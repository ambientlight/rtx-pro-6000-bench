"""Offline launcher for the pinned 1017 checkpoint on upstream SGLang.

Startup smokes and signal/log forwarding are retained from the deployed
ambientlight/sglang overlay 91051e1, based on 0xSero 45f538a.
"""
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL = Path(os.environ.get('MODEL_PATH','/models/DeepSeek-V4.1-Flash'))
STATE = Path(os.environ.get('STATE_PATH','/state'))
PORT = int(os.environ.get('SERVER_PORT','18000'))

def served_model_name():
    value = os.environ.get('SERVED_MODEL_NAME', 'deepseek-v4-flash').strip()
    assert value, 'SERVED_MODEL_NAME must not be empty'
    return value

def require_api_key():
    value = os.environ.get('REQUIRE_API_KEY', '0')
    assert value in ('0', '1'), 'REQUIRE_API_KEY must be 0 or 1'
    return value == '1'

def save(name, value):
    if os.environ.get('CONTENT_FREE_LOGGING') == '1' and name.startswith('smoke'):
        value = {'status': 'passed'}
    STATE.mkdir(parents=True,exist_ok=True)
    temporary = STATE/(name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(STATE/name)

def file_identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def prepare():
    """Accept the existing verified inode, or rehash a relocated immutable file."""
    receipt = json.loads(Path(__file__).with_name('checkpoint.json').read_text())
    if receipt['revision'] != REVISION:
        raise ValueError('unexpected checkpoint revision')
    root = MODEL.resolve(strict=True)
    names = set()
    for entry in receipt['files']:
        name = entry['name']
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or name in names:
            raise ValueError('invalid checkpoint manifest')
        names.add(name)
        st = path.stat()
        if st.st_size != entry['bytes']:
            raise ValueError(f'checkpoint size mismatch in {name}')
        if file_identity(st) == (entry['device'], entry['inode'], entry['bytes'],
                                 entry['mtime_ns'], entry['ctime_ns']):
            continue
        # LFS stores a SHA256 of the content. Ordinary git blobs include a header.
        digest = hashlib.sha256() if len(entry['digest']) == 64 else hashlib.sha1()
        if len(entry['digest']) == 40:
            digest.update(f'blob {st.st_size}\0'.encode())
        with path.open('rb') as f:
            while block := f.read(8 << 20):
                digest.update(block)
        after = path.stat()
        if (digest.hexdigest() != entry['digest'] or
                file_identity(st) != file_identity(after)):
            raise ValueError(f'checkpoint content changed in {name}')
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values()) <= names:
        raise ValueError('checkpoint manifest lacks a tensor shard')


def key():
    STATE.mkdir(parents=True,exist_ok=True)
    path = STATE/'api-key'
    value = os.environ.get('API_KEY','').strip()
    if value:
        path.write_text(value)
        path.chmod(0o600)
    elif not path.exists():
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as handle:
            handle.write(secrets.token_urlsafe(32))
    return path.read_text().strip()

def request(path,payload=None,timeout=10):
    headers = {'Content-Type':'application/json'}
    if require_api_key():
        headers['Authorization'] = 'Bearer '+key()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}'+path,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req,timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {'status':response.status}

def smoke():
    result = request('/v1/chat/completions',dict(model=served_model_name(),temperature=0,
        chat_template_kwargs={'thinking':False},messages=[dict(role='user',
        content='What is 19 + 23? Reply only with the number.')]),timeout=300)
    choice = result['choices'][0]
    assert choice['message']['content'].strip() == '42' and choice['finish_reason'] == 'stop', result
    save('smoke.json',result)
    print('Fresh inference passed: 19 + 23 = 42',flush=True)
    common = dict(model=served_model_name(),temperature=0,
                  chat_template_kwargs={'thinking':False})
    structured = request('/v1/chat/completions',dict(common,
        messages=[dict(role='user',content='Return an object whose answer is the integer 42.')],
        response_format={'type':'json_schema','json_schema':{'name':'answer','strict':True,
            'schema':{'type':'object','properties':{'answer':{'type':'integer'}},
                      'required':['answer'],'additionalProperties':False}}}),timeout=300)
    assert json.loads(structured['choices'][0]['message']['content']) == {'answer':42}
    save('smoke-structured.json',structured)
    tool = {'type':'function','function':{'name':'lookup_fixture',
        'description':'Retrieve a stored test value.',
        'parameters':{'type':'object','properties':{'key':{'type':'string'}},
                      'required':['key'],'additionalProperties':False}}}
    messages = [dict(role='user',content='Use lookup_fixture to retrieve the value for key alpha. Do not guess.')]
    called = request('/v1/chat/completions',dict(common,messages=messages,tools=[tool]),timeout=300)
    assistant = called['choices'][0]['message']
    calls = assistant.get('tool_calls') or []
    assert len(calls)==1 and calls[0]['function']['name']=='lookup_fixture', called
    assert json.loads(calls[0]['function']['arguments']) == {'key':'alpha'}
    messages += [assistant,dict(role='tool',tool_call_id=calls[0]['id'],content='{"value":42}')]
    continued = request('/v1/chat/completions',dict(common,messages=messages,tools=[tool]),timeout=300)
    assert '42' in continued['choices'][0]['message']['content'], continued
    save('smoke-tools.json',{'call':called,'continuation':continued})
    from PIL import Image,ImageDraw
    image = Image.new('RGB',(3024,588),'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((200,100,588,488),fill='red')
    draw.rectangle((2400,100,2788,488),fill='blue')
    buffer = io.BytesIO()
    image.save(buffer,format='PNG')
    visual = request('/v1/chat/completions',dict(common,messages=[dict(role='user',content=[
        {'type':'text','text':'Describe the two colored shapes and their left-to-right order. Be concise.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}}
    ])]),timeout=300)
    text = visual['choices'][0]['message']['content'].lower()
    assert all(word in text for word in ('red','circle','blue','square')), visual
    assert visual['usage']['prompt_tokens_details']['image_tokens'] == 1024, visual
    save('smoke-vision.json',visual)
    print('Structured output, tool round trip and full-budget native vision passed.',flush=True)

def _write_log_bytes(output, data):
    """Binary sinks may accept partial writes; never silently drop the remainder."""
    remaining = memoryview(data)
    while remaining:
        written = output.write(remaining)
        if written is None or written <= 0:
            raise OSError('Log sink stopped accepting bytes')
        remaining = remaining[written:]
    if data:
        output.flush()


def forward_log_bytes(source, output, secret, chunk_size=65536):
    """Drain raw bytes, preserving Unicode and redacting keys across read boundaries.

    A shared, multi-process stdout stream need not be valid UTF-8 even when its
    individual writers are. Never decode it in the only pipe reader. Retain only
    a possible secret prefix, not an entire (potentially multi-MB) log line.
    Popen uses bufsize=0 so read() forwards available bytes without waiting for
    a full chunk or newline. Docker remains responsible for log framing.
    """
    secret = secret.encode('utf-8')
    pending = b''
    while True:
        block = source.read(chunk_size)
        if not block:
            _write_log_bytes(output, pending)
            return
        data = pending + block
        if not secret:
            _write_log_bytes(output, data)
            continue
        pieces, position = [], 0
        while True:
            found = data.find(secret, position)
            if found < 0:
                break
            pieces.extend((data[position:found], b'[REDACTED]'))
            position = found + len(secret)
        remainder = data[position:]
        keep = min(len(secret) - 1, len(remainder))
        while keep and not remainder.endswith(secret[:keep]):
            keep -= 1
        pieces.append(remainder[:-keep] if keep else remainder)
        pending = remainder[-keep:] if keep else b''
        _write_log_bytes(output, b''.join(pieces))


class ServerLogForwarder:
    """A dead reader must terminate the server, never strand a live unread pipe."""

    def __init__(self, process, secret, output=None, shutdown_timeout=10, kill_timeout=5):
        self.process = process
        self.secret = secret
        self.output = sys.stdout.buffer if output is None else output
        self.shutdown_timeout = shutdown_timeout
        self.kill_timeout = kill_timeout
        self.failed = threading.Event()
        self.failure_kind = None
        self.stop_requested = False
        self._stop_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='dsv41-log-forwarder', daemon=True)

    def start(self):
        self.thread.start()

    def _signal(self, signum):
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass

    def request_stop(self, *_):
        # Signal handlers must not wait on the reader or acquire its stop lock.
        self.stop_requested = True
        if self.process.poll() is None:
            self._signal(signal.SIGTERM)

    def stop(self):
        with self._stop_lock:
            if self.process.poll() is not None:
                return
            self._signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                try:
                    self.process.wait(timeout=self.kill_timeout)
                except subprocess.TimeoutExpired:
                    # TimeoutExpired includes process.args, which may have a key.
                    raise RuntimeError('Server did not exit after bounded SIGKILL wait') from None

    def _run(self):
        try:
            forward_log_bytes(self.process.stdout, self.output, self.secret)
            # EOF just before normal process exit is fine. EOF while the server
            # remains alive is a broken logging path and must fail closed.
            self.process.wait(timeout=1)
        except BaseException as exc:
            # Do not retain/print exception text: it can contain captured secrets.
            self.failure_kind = type(exc).__name__
            self.failed.set()
            try:
                self.process.stdout.close()
            except BaseException:
                pass
            # Also covers failures during startup/smoke while the main thread
            # is inside an HTTP call. Never expose the original log exception
            # through threading.excepthook if cleanup itself fails.
            try:
                self.stop()
            except BaseException:
                pass  # The main thread independently checks failure and stops.

    def check(self):
        if self.failed.is_set():
            raise RuntimeError(f'Server log forwarding failed ({self.failure_kind}); refusing an unread live pipe')

    def wait(self):
        while True:
            self.check()
            if self.stop_requested:
                self.stop()
            try:
                result = self.process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
        self.thread.join(timeout=5)
        self.check()
        if self.thread.is_alive():
            raise RuntimeError('Server exited but log forwarding did not finish within 5 seconds')
        return result


def server_arguments():
    args = [
        '--model-path', str(MODEL), '--served-model-name', served_model_name(),
        '--trust-remote-code', '--load-format', 'safetensors',
        '--tp-size', '4', '--ep-size', '4', '--attention-backend', 'dsv4',
        '--moe-runner-backend', 'flashinfer_mxfp4',
        '--engram-config', '{"placement":"cpu","replication":"none","lookup":"all_reduce"}',
        '--context-length', '524288', '--mem-fraction-static', '0.87',
        '--max-running-requests', '8', '--chunked-prefill-size', '2048',
        '--prefill-decode-interval', '4', '--min-free-slots-delay', '1',
        '--cuda-graph-max-bs-decode', '8', '--enable-decoder-swa-bounded-replay',
        '--speculative-algorithm', 'DSPARK', '--speculative-dspark-block-size', '5',
        '--tool-call-parser', 'deepseekv41', '--reasoning-parser', 'deepseek-v41',
        '--preferred-sampling-params', '{"temperature":1.0,"top_p":0.95,"max_new_tokens":32768}',
        '--random-seed', '0', '--enable-metrics',
        '--host', os.environ.get('SGLANG_LISTEN_ADDRESS','127.0.0.1'), '--port', str(PORT),
    ]
    if os.environ.get('PEARL_WOB_MODULE'):
        if os.environ.get('PEARL_PUBLIC_REPLAY') != '1':
            raise ValueError('inference proof generation requires public replay inputs')
        args += ['--cuda-graph-backend-decode', 'disabled', '--cuda-graph-backend-prefill', 'disabled']
    return args


def serve():
    args = server_arguments()
    authenticated = require_api_key()
    save('launch.json', {'revision': REVISION, 'args': args, 'require_api_key': authenticated})
    secret = key() if authenticated else ''
    env = os.environ.copy()
    command = [sys.executable,'-m','sglang.launch_server',*args]
    if authenticated:
        command.extend(['--api-key',secret])
    process = subprocess.Popen(command,
        env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=True)
    forwarder = ServerLogForwarder(process, secret)
    signal.signal(signal.SIGTERM,forwarder.request_stop)
    signal.signal(signal.SIGINT,forwarder.request_stop)
    forwarder.start()
    try:
        deadline = time.monotonic()+1800
        while process.poll() is None and time.monotonic()<deadline:
            forwarder.check()
            if forwarder.stop_requested:
                raise RuntimeError('Server startup interrupted')
            try:
                request('/health',timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            forwarder.check()
            raise RuntimeError('Server failed to become healthy; inspect container logs')
        smoke()
        forwarder.check()
        auth_description = 'API key required' if authenticated else 'no backend API key (legacy compatibility)'
        print(f'Ready: {served_model_name()} on port {PORT}; {auth_description}.',flush=True)
        return forwarder.wait()
    finally:
        forwarder.stop()
        forwarder.thread.join(timeout=1)

if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv)>1 else 'run'
    if command == 'args': print(json.dumps(server_arguments()))
    elif command == 'health': request('/health')
    elif command == 'smoke': smoke()
    elif command == 'prepare': prepare()
    elif command == 'run':
        prepare()
        sys.exit(serve())
    else: raise SystemExit('Commands: run, prepare, health, smoke, args')
