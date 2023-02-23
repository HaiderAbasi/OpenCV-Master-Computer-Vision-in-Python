@echo off

set "required_os=Windows 10"
set "required_python_version=3.8.10"

python -c "import platform; print(platform.python_version())" > version.txt
set /p python_version=<version.txt
del version.txt

echo testing "%python_version%"
echo testing2 "%required_python_version%"

if not "%python_version%" == "%required_python_version%" (
  echo Error: This batch file requires Python version %required_python_version%
  
  echo Installing Python 3.8.10

  curl https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe -o python-3.8.10-amd64.exe

  start /wait python-3.8.10-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

  echo Python 3.8.10 installation complete

  where python
  if %errorlevel% equ 0 (
    echo Python is installed
    del python-3.8.10-amd64.exe
    echo Setup file deleted
  ) else (
    echo Python is not installed
    goto end
  )
)

echo testing done

ver > os.txt
set /p os=<os.txt
del os.txt

for /f "tokens=4,5" %%i in ('systeminfo ^| findstr /c:"OS Name" /c:"OS Version"') do (
  if "%%i" == "Windows" (
    if "%%j" == "10" (
        echo Running on Windows 10
        goto setup
        )
  ) 
  else (
    echo Error: This batch file requires operating system Windows 10
    exit /b
  )
)

:setup
echo Setting up DeepSort_YoloV5 on your system!
rem Create virtual environment using virtualenv
virtualenv ds_env

echo Virtual environment created successfully!

rem Activate the virtual environment & Install modules listed in requirements.txt
call ds_env/Scripts/activate && pip install -r requirements.txt && python utils/correct_torch_call.py

echo Modules installed successfully!


pause

:end