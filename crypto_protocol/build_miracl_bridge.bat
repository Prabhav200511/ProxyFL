@echo off
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" exit /b 1
call "%VCVARS%" >nul || exit /b 1
set "ROOT=%~dp0"
python -B "%ROOT%prepare_miracl_bridge.py" || exit /b 1
cl /nologo /O2 /LD /Fo"%ROOT%build\\" /I"%ROOT%build" ^
  "%ROOT%miracl_core_bridge.c" "%ROOT%build\hash.c" ^
  "%ROOT%build\aes.c" "%ROOT%build\gcm.c" ^
  /Fe:"%ROOT%miracl_core.dll" /link /IMPLIB:"%ROOT%build\miracl_core.lib"
