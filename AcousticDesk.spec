# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[('frontend', 'frontend'), ('backend', 'backend')],
    hiddenimports=['uvicorn', 'fastapi', 'sounddevice', 'sklearn', 'scipy', 'numpy', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AcousticDesk',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AcousticDesk',
)

app = BUNDLE(
    coll,
    name='AcousticDesk.app',
    icon=None,
    bundle_identifier='com.acousticdesk.buttons',
    info_plist={
        'NSMicrophoneUsageDescription': 'AcousticDesk requires microphone access to detect table taps.',
        'NSCameraUsageDescription': 'AcousticDesk requires camera access for multi-modal spatial tap tracking.',
        'LSBackgroundOnly': '0',
    }
)
