@echo off
setlocal
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" exit /b 1
call "%VCVARS%" >nul || exit /b 1
set "ROOT=%~dp0"
if not exist "%ROOT%build" mkdir "%ROOT%build" || exit /b 1
cl /nologo /O2 /LD /Fo"%ROOT%build\\" /I"%ROOT%..\core-master\c" ^
  "%ROOT%miracl_core_bridge.c" "%ROOT%..\core-master\c\hash.c" ^
  "%ROOT%..\core-master\c\aes.c" "%ROOT%..\core-master\c\gcm.c" ^
  /Fe:"%ROOT%miracl_core.dll"
