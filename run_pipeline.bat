@echo off

echo [STAGE 0] Cleaning stale artifacts...
if exist configs\rag_generated_config.json del configs\rag_generated_config.json
if exist output\*.png del /q output\*.png
if exist output\images\*.png del /q output\images\*.png
if exist output\latest_annotations.json del /q output\latest_annotations.json

echo [STAGE 1] Running RAG query engine...
call venv_gpu\Scripts\activate.bat
python rag_query_engine.py %*
if %errorlevel% neq 0 (
    echo [ERROR] RAG stage failed
    exit /b 1
)

:: SAFETY NET: Force the generated config to only render 1 scene
powershell -Command "(Get-Content -Path 'configs\rag_generated_config.json') -replace '\"num_scenes\": \d+', '\"num_scenes\": 1' | Set-Content -Path 'configs\rag_generated_config.json'"

echo [STAGE 1] Config written and capped to 1 scene. Clearing Python heap...
echo.

echo [STAGE 2] Running BlenderProc in clean CPU process...
set OIIO_CACHE_MEMORY_MB=2048
set BLENDER_CYCLES_TEXTURE_LIMIT=2048
set MALLOC_TRIM_THRESHOLD_=65536

"C:\Users\priya\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\blenderproc.exe" run blender_scenes\scripts\indian_street_gen.py -- ^
    --config configs\rag_generated_config.json ^
    --output_dir output
if %errorlevel% neq 0 (
    echo [ERROR] BlenderProc failed
    exit /b 1
)
echo [STAGE 2] Render complete.
echo.

echo [STAGE 3] Running YOLO Vision Inference
echo Labeling...
python run_yolo_stage.py
if %errorlevel% neq 0 (
    echo [ERROR] YOLO inference stage failed
    exit /b 1
)

echo.
echo [DONE] Full pipeline finished perfectly!