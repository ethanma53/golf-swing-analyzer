from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from analyzer import analyze_swing, VideoProcessingError, NoPoseDetectedError, LowConfidenceError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an mp4 or mov file.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        result = analyze_swing(temp_path)

        return {
            "frames_analyzed": len(result['frames_data']),
            "avg_confidence": result['avg_confidence'],
            "video_info": result['video_info'],
            "feedback": ["Video processed successfully. Metrics not yet implemented."]
        }
    except VideoProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NoPoseDetectedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LowConfidenceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
