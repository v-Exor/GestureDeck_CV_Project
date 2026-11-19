"""Read-only Windows foreground check; never activates or launches PowerPoint."""

import os


def powerpoint_slideshow_focused() -> bool:
    return focused_powerpoint_window() is not None


def focused_powerpoint_window() -> int | None:
    """Validated foreground slideshow handle for optional window-only recording."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    window = user32.GetForegroundWindow()
    name = ctypes.create_unicode_buffer(256)
    if not window or not user32.GetClassNameW(window, name, len(name)):
        return None
    if name.value.casefold() != "screenclass":
        return None
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
    process = kernel32.OpenProcess(0x1000, False, pid.value)
    if not process:
        return None
    try:
        size = wintypes.DWORD(32768)
        path = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
            return None
        return window if os.path.basename(path.value).casefold() == "powerpnt.exe" else None
    finally:
        kernel32.CloseHandle(process)


def powerpoint_window_bounds(window):
    """Physical screen bounds of a validated window; never changes its focus."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    rectangle = wintypes.RECT()
    if not user32.GetWindowRect(window, ctypes.byref(rectangle)):
        raise OSError("Cannot read PowerPoint window bounds")
    bounds = rectangle.left, rectangle.top, rectangle.right, rectangle.bottom
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise OSError("PowerPoint window has empty bounds")
    return bounds
