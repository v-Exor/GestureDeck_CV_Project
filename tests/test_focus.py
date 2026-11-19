import ntpath
from types import SimpleNamespace
import pytest
import gesturedeck.focus as focus


class Function:
    def __init__(self, call): self.call = call
    def __call__(self, *args): return self.call(*args)


@pytest.mark.parametrize('class_name,path,opened,query_ok,expected', [
    ('screenClass', r'C:\Office\POWERPNT.EXE', True, True, True),
    ('screenClass', r'C:\Office\powerpnt.exe', True, True, True),
    ('PPTFrameClass', r'C:\Office\POWERPNT.EXE', True, True, False),
    ('screenClass', r'C:\Other\app.exe', True, True, False),
    ('screenClass', r'C:\Office\POWERPNT.EXE', False, True, False),
    ('screenClass', r'C:\Office\POWERPNT.EXE', True, False, False),
])
def test_native_focus_checks_window_class_and_process_and_closes_handle(
        monkeypatch, class_name, path, opened, query_ok, expected):
    import ctypes
    closed = []
    def name(_, buffer, __):
        buffer.value = class_name
        return len(class_name)
    def pid(_, ref):
        ref._obj.value = 42
        return 1
    def image(_, __, buffer, ___):
        buffer.value = path
        return query_ok
    user32 = SimpleNamespace(GetForegroundWindow=Function(lambda: 100),
        GetClassNameW=Function(name), GetWindowThreadProcessId=Function(pid))
    kernel32 = SimpleNamespace(OpenProcess=Function(lambda *_: 200 if opened else 0),
        QueryFullProcessImageNameW=Function(image), CloseHandle=Function(closed.append))
    monkeypatch.setattr(focus, 'os', SimpleNamespace(name='nt', path=ntpath))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **_: user32 if name=='user32' else kernel32,
                        raising=False)
    assert focus.powerpoint_slideshow_focused() is expected
    assert closed == ([200] if opened and class_name=='screenClass' else [])


def test_unsupported_platform_fails_closed(monkeypatch):
    monkeypatch.setattr(focus, 'os', SimpleNamespace(name='posix'))
    assert not focus.powerpoint_slideshow_focused()
