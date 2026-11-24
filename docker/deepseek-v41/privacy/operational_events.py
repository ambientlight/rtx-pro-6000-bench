"""Extract only named numeric scheduler statistics, never arbitrary log text.

The source and function must match the installed scheduler. The entire message
must match its numeric grammar; unfamiliar fields fail closed. Nothing calls
str(message), formats arguments, reads locals, or copies user-supplied labels.
"""
import math
import re

SCHEDULER_SOURCES = frozenset(
    root + 'sglang/srt/managers/scheduler_components/metrics_reporter.py'
    for root in ('/sgl-workspace/sglang/python/', '/opt/sglang/python/')
)
NUMBER = r'(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,6})?'
# Original field -> stored field, value type, upper bound, displayed label/unit.
FIELDS = {
    '#new-seq': ('new_requests', 'int', 10**9, 'new requests', ''),
    '#new-token': ('input_tokens', 'int', 10**12 - 1, 'input', ' tokens'),
    '#cached-token': ('cached_tokens', 'int', 10**12 - 1, 'cached', ' tokens'),
    '#running-req': ('running', 'int', 10**9, 'running', ''),
    '#queue-req': ('queued', 'int', 10**9, 'queued', ''),
    '#pending-token': ('pending_tokens', 'int', 10**12 - 1, 'pending', ' tokens'),
    '#replay-token': ('replay_tokens', 'int', 10**12 - 1, 'replayed', ' tokens'),
    'input throughput (token/s)': ('input_tps', 'float', 10**12 - 1, 'input rate', ' tokens/s'),
    'gen throughput (token/s)': ('output_tps', 'float', 10**12 - 1, 'output rate', ' tokens/s'),
    'token usage': ('kv_usage', 'ratio', 1, 'KV cache', '%'),
    'full token usage': ('full_kv_usage', 'ratio', 1, 'full KV cache', '%'),
    'swa token usage': ('swa_usage', 'ratio', 1, 'sliding-window cache', '%'),
    'mamba usage': ('mamba_usage', 'ratio', 1, 'Mamba cache', '%'),
    'gpu token usage': ('gpu_usage', 'ratio', 1, 'GPU token pool', '%'),
    'cpu token usage': ('cpu_usage', 'ratio', 1, 'CPU token pool', '%'),
    '#token': ('kv_tokens', 'int', 10**12 - 1, 'KV pool', ' tokens'),
    '#full token': ('full_kv_tokens', 'int', 10**12 - 1, 'full KV pool', ' tokens'),
    '#swa token': ('swa_tokens', 'int', 10**12 - 1, 'sliding-window pool', ' tokens'),
    'mamba num': ('mamba_count', 'int', 10**12 - 1, 'Mamba slots', ''),
    '#gpu token': ('gpu_tokens', 'int', 10**12 - 1, 'GPU pool', ' tokens'),
    '#cpu token': ('cpu_tokens', 'int', 10**12 - 1, 'CPU pool', ' tokens'),
    'cuda graph': ('cuda_graph', 'bool', 1, 'CUDA graph', ''),
    'cpu graph': ('cpu_graph', 'bool', 1, 'CPU graph', ''),
    'npu graph': ('npu_graph', 'bool', 1, 'NPU graph', ''),
    'musa graph': ('musa_graph', 'bool', 1, 'MUSA graph', ''),
    'accept len': ('accept_length', 'float', 10**6, 'draft acceptance', ' tokens/step'),
    'accept rate': ('accept_rate', 'ratio', 1, 'draft acceptance rate', '%'),
    'cap len': ('cap_length', 'float', 10**6, 'draft cap', ' tokens'),
    'block accept len': ('block_accept_length', 'float', 10**6, 'block acceptance', ' tokens'),
    '#bootstrap-req': ('bootstrap', 'int', 10**9, 'bootstrapping', ''),
    '#inflight-req': ('inflight', 'int', 10**9, 'in flight', ''),
    '#optimistic-req': ('optimistic', 'int', 10**9, 'optimistic', ''),
    'waiting-image-req': ('waiting_image', 'int', 10**9, 'waiting for images', ''),
    '#prealloc-req': ('prealloc', 'int', 10**9, 'preallocating', ''),
    '#transfer-req': ('transfer', 'int', 10**9, 'transferring', ''),
    '#retracted-req': ('retracted', 'int', 10**9, 'retracted', ''),
    'pre-allocated usage': ('prealloc_usage', 'ratio', 1, 'preallocated cache', '%'),
    'fwd occupancy': ('forward_occupancy', 'percent', 100, 'forward occupancy', '%'),
    'est. prefill TFLOPS/s (per GPU)': ('prefill_tflops', 'float', 10**9, 'estimated prefill', ' TFLOPS/s/GPU'),
    'est. decode TFLOPS/s (per GPU)': ('decode_tflops', 'float', 10**9, 'estimated decode', ' TFLOPS/s/GPU'),
    'est. read BW (GB/s per GPU)': ('read_bandwidth', 'float', 10**9, 'estimated reads', ' GB/s/GPU'),
    'est. write BW (GB/s per GPU)': ('write_bandwidth', 'float', 10**9, 'estimated writes', ' GB/s/GPU'),
}
BY_KEY = {spec[0]: spec[1:] for spec in FIELDS.values()}
REQUIRED = {
    'prefill': frozenset(('new_requests', 'input_tokens', 'cached_tokens', 'running', 'queued', 'pending_tokens', 'input_tps')),
    'decode': frozenset(('running', 'queued', 'output_tps')),
}
ESTIMATE = re.compile(r'est uncap acc len: (' + NUMBER + r') \[(' + NUMBER + r'), (' + NUMBER + r')\](?=, |$)')
PART = re.compile(r'([^:\n]{1,70}): (' + NUMBER + r'%?|True|False)(?=, |$)')


def clean_metrics(value):
    """Validate again at the publisher; JSON types and names are closed."""
    if type(value) is not dict or type(value.get('phase')) is not str or value['phase'] not in REQUIRED:
        return None
    phase = value['phase']
    if set(value) - ({'phase'} | BY_KEY.keys()) or not REQUIRED[phase] <= value.keys():
        return None
    result = {'phase': phase}
    for key, number in value.items():
        if key == 'phase':
            continue
        kind, upper, _label, _unit = BY_KEY[key]
        if kind == 'bool':
            valid = type(number) is bool
        else:
            valid = (type(number) is int if kind == 'int' else type(number) in (int, float))
            valid = valid and 0 <= number <= upper and math.isfinite(number)
        if not valid:
            return None
        result[key] = number
    return result


def scheduler_metrics(source, function, message, args):
    if (type(source) is not str or source not in SCHEDULER_SOURCES
            or type(function) is not str or function not in ('report_prefill_stats', 'report_decode_stats')
            or type(message) is not str or len(message) > 4096
            or type(args) is not tuple or len(args) != 0):
        return None
    phase = 'prefill' if function == 'report_prefill_stats' else 'decode'
    prefix = re.match(('Prefill' if phase == 'prefill' else 'Decode') + r' batch(?: \[[0-9]{1,12}\])?, ', message)
    if prefix is None:
        return None
    values, position, seen_estimate = {'phase': phase}, prefix.end(), False
    while position < len(message):
        # This optional estimator adds a numeric interval with an embedded comma.
        # Validate and discard it; the ordinary acceptance metrics are sufficient.
        estimate = ESTIMATE.match(message, position)
        if estimate is not None:
            numbers = [float(v) for v in estimate.groups()]
            if seen_estimate or not 0 <= numbers[1] <= numbers[0] <= numbers[2] <= 10**6:
                return None
            seen_estimate, end = True, estimate.end()
        else:
            part = PART.match(message, position)
            if part is None or part[1] not in FIELDS:
                return None
            key, kind, _upper, _label, _unit = FIELDS[part[1]]
            if key in values:
                return None
            value = part[2]
            if kind == 'bool':
                if value not in ('True', 'False'):
                    return None
                number = value == 'True'
            elif kind == 'int':
                if not value.isascii() or not value.isdigit():
                    return None
                number = int(value)
            else:
                if value.endswith('%') != (kind == 'percent') or value in ('True', 'False'):
                    return None
                number = float(value.rstrip('%'))
            values[key], end = number, part.end()
        if end == len(message):
            return clean_metrics(values)
        position = end + 2
    return None


def describe_metrics(value):
    values = clean_metrics(value)
    if values is None:
        return None
    order = ['running', 'queued', 'new_requests', 'input_tokens', 'cached_tokens',
             'input_tps', 'output_tps', 'full_kv_usage', 'swa_usage', 'kv_usage',
             'accept_length', 'accept_rate']
    order += [key for key in BY_KEY if key not in order]
    parts = []
    for key in order:
        if key not in values:
            continue
        number = values[key]
        kind, _upper, label, unit = BY_KEY[key]
        if kind == 'bool':
            text = 'on' if number else 'off'
        elif kind == 'int':
            text = str(number)
        elif kind == 'ratio':
            text = f'{number * 100:.0f}'
        else:
            text = f'{number:.2f}'
        parts.append(label + '=' + text + unit)
    return values['phase'].capitalize() + ': ' + ', '.join(parts)
