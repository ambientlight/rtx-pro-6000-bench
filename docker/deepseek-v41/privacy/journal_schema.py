"""Closed schema for diagnostics allowed into Docker and the host journal."""
import json
import math
import re

from content_free_logging import _ERRORS, _frame, _number
from operational_events import BY_KEY, NUMBER, SCHEDULER_SOURCES, clean_metrics, describe_metrics

MAX_DATAGRAM = 60 * 1024
EVENTS = frozenset(('privacy_started', 'server_starting', 'proxy_starting', 'log',
                    'unhandled_exception', 'stack_dump', 'stack_dump_failed',
                    'log_publisher_started', 'log_publisher_exit', 'log_publisher_failed'))


def frame(value):
    source = value.get('source')
    return _frame(source if isinstance(source, str) and len(source) <= 240 else 'omitted',
                  _number(value.get('line')))


def clean(raw):
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if (not isinstance(value, dict) or not isinstance(value.get('event'), str)
            or value['event'] not in EVENTS):
        return None
    at = value.get('time')
    if type(at) not in (int, float) or not 0 <= at < 253402300800 or not math.isfinite(at):
        return None
    event = {'time': at, 'pid': _number(value.get('pid')), 'event': value['event']}
    if value['event'] == 'log':
        level = value.get('level')
        event['level'] = level if type(level) is int and level in (0, 10, 20, 30, 40, 50) else 0
        event.update(frame(value))
        if event['level'] == 20 and event['source'] in SCHEDULER_SOURCES:
            metrics = clean_metrics(value.get('metrics'))
            if metrics is not None:
                event['metrics'] = metrics
    if isinstance(value.get('exceptions'), list):
        event['exceptions'] = []
        for item in value['exceptions'][:4]:
            if not isinstance(item, dict):
                continue
            kind = item.get('type')
            frames = item.get('frames')
            event['exceptions'].append({
                'type': kind if isinstance(kind, str) and kind in _ERRORS else 'OtherError',
                'frames': [frame(f)
                           for f in (frames[-8:] if isinstance(frames, list) else [])
                           if isinstance(f, dict)]})
    if value['event'] in ('stack_dump', 'stack_dump_failed'):
        event['target_pid'] = _number(value.get('target_pid'))
        event['native'] = value.get('native') is True
        threads = value.get('threads')
        if isinstance(threads, list):
            event['thread_count'] = len(threads)
            event['threads'] = []
            for item in threads[:4]:
                if not isinstance(item, dict):
                    continue
                frames = item.get('frames')
                event['threads'].append({
                    'os_thread_id': _number(item.get('os_thread_id')),
                    'frames': [frame(f)
                               for f in (frames[-8:] if isinstance(frames, list) else [])
                               if isinstance(f, dict)]})
    if value['event'] in ('stack_dump', 'stack_dump_failed', 'log_publisher_exit'):
        code = value.get('exit_code')
        if type(code) is int and -255 <= code <= 255:
            event['exit_code'] = code
    return event


LIFECYCLE = {
    'privacy_started': 'Privacy logging active',
    'server_starting': 'Inference server starting',
    'proxy_starting': 'LiteLLM proxy starting',
    'log_publisher_started': 'Log publisher started',
    'log_publisher_failed': 'Log publisher failed',
    'unhandled_exception': 'Unhandled exception',
    'stack_dump_failed': 'Stack capture failed',
}
LEVELS = {0: 'LOG', 10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}


def location(value):
    source = value.get('source', 'omitted')
    if (type(source) is str and source != 'omitted'
            and re.fullmatch(r'/[A-Za-z0-9_./+\-]{1,240}', source)):
        return source.rsplit('/', 1)[-1] + ':' + str(_number(value.get('line')))
    return 'location unavailable'


def encode(event):
    """Human-readable text built exclusively from the closed event schema."""
    error = (event.get('level', 0) >= 40 or event['event'] in
             ('unhandled_exception', 'stack_dump_failed', 'log_publisher_failed') or
             (event['event'] == 'log_publisher_exit' and event.get('exit_code', 0) != 0))
    level = LEVELS.get(event.get('level', 40 if error else 20), 'LOG')
    kind = event['event']
    if kind == 'log':
        body = describe_metrics(event.get('metrics')) or 'Diagnostic event (message omitted)'
    elif kind == 'log_publisher_exit':
        body = 'Application exited; code=' + str(event.get('exit_code', 0))
    elif kind == 'stack_dump':
        body = ('Stack capture: target=' + str(_number(event.get('target_pid')))
                + ', threads=' + str(_number(event.get('thread_count')))
                + ', native=' + ('yes' if event.get('native') else 'no'))
    else:
        body = LIFECYCLE.get(kind)
        if body is None:
            return None
    exceptions = []
    for item in event.get('exceptions', [])[:2]:
        name = item.get('type')
        name = name if name in _ERRORS else 'OtherError'
        sites = [location(f) for f in item.get('frames', [])[-3:]]
        sites = [s for s in sites if s != 'location unavailable']
        exceptions.append(name + (' at ' + ' -> '.join(sites) if sites else ''))
    if exceptions:
        body += '; ' + '; caused by '.join(exceptions) + ' (exception text omitted)'
    suffix = ' [worker ' + str(_number(event.get('pid')))
    if kind == 'log':
        suffix += '; ' + location(event)
    payload = (level + ' ' + body + suffix + ']\n').encode('ascii')
    # Bounded plain text fits one Linux PIPE_BUF write. Full structured traces
    # remain in the private files, including stack_dump's thread locations.
    if len(payload) > 3500:
        return None
    return (2 if error else 1), payload


def classify_line(line):
    """Validate the complete public text grammar for deployment smoke checks."""
    match = re.fullmatch(r'(LOG|DEBUG|INFO|WARNING|ERROR|CRITICAL) (.+) \[worker [0-9]{1,19}'
                         r'(?:; (?:[A-Za-z0-9_.+\-]{1,240}:[0-9]{1,19}|location unavailable))?\]', line)
    if match is None:
        return None
    body = match[2]
    exceptions, metrics = [], None
    if body.endswith(' (exception text omitted)'):
        body = body[:-len(' (exception text omitted)')]
        body, separator, details = body.partition('; ')
        if not separator:
            return None
        for item in details.split('; caused by '):
            name, _, sites = item.partition(' at ')
            if name not in _ERRORS | {'OtherError'}:
                return None
            if sites and not re.fullmatch(r'[A-Za-z0-9_.+\-]{1,240}:[0-9]{1,19}'
                                         r'(?: -> [A-Za-z0-9_.+\-]{1,240}:[0-9]{1,19}){0,2}', sites):
                return None
            exceptions.append({'type': name})
    if body in LIFECYCLE.values():
        kind = next(key for key, text in LIFECYCLE.items() if text == body)
    elif body == 'Diagnostic event (message omitted)':
        kind = 'log'
    elif re.fullmatch(r'Application exited; code=-?[0-9]{1,3}', body):
        kind = 'log_publisher_exit'
    elif re.fullmatch(r'Stack capture: target=[0-9]{1,19}, threads=[0-9]{1,19}, native=(yes|no)', body):
        kind = 'stack_dump'
    elif body.startswith(('Prefill: ', 'Decode: ')):
        phase, fields = body.split(': ', 1)
        labels = {spec[2]: (key, *spec) for key, spec in BY_KEY.items()}
        values = {'phase': phase.lower()}
        for part in fields.split(', '):
            label, separator, text = part.partition('=')
            if not separator or label not in labels:
                return None
            key, value_type, _upper, _label, unit = labels[label]
            if key in values or (unit and not text.endswith(unit)):
                return None
            text = text[:-len(unit)] if unit else text
            if value_type == 'bool':
                if text not in ('on', 'off'):
                    return None
                number = text == 'on'
            else:
                if not re.fullmatch(NUMBER, text):
                    return None
                if value_type == 'int':
                    if '.' in text:
                        return None
                    number = int(text)
                else:
                    number = float(text) / (100 if value_type == 'ratio' else 1)
            values[key] = number
        if clean_metrics(values) is None:
            return None
        metrics = values
        kind = 'log'
    else:
        return None
    result = {'event': kind, 'exceptions': exceptions, 'readable': True}
    if metrics is not None:
        result['metrics'] = metrics
    return result
