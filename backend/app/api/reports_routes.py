from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import os
import logging
import subprocess
import pandas as pd

router = APIRouter()
logging.basicConfig(level=logging.INFO)

REPORTS_DIR = "reports"  # Make sure this folder exists
FULL_REPORT_FILENAME = "full_report.html"
FULL_REPORT_PDF = "full_report.pdf"
FORECAST_LOG_FILENAME = "forecasting_log.txt"
AI_INSIGHTS_FILENAME = "ai_insights_report.txt"

@router.post("/generate-full")
async def generate_full_report(
    dataset_id: str = Query(..., description="The dataset ID for this analysis")
):
    """
    Generate a comprehensive full report in HTML format.
    For simplicity, we assume your pipeline saves:
      - EDA to:    reports/eda_report_{dataset_id}.txt
      - Training to: reports/training_results_{dataset_id}.csv (or .txt)
      - Forecast log: forecasting_log.txt
      - AI insights:  ai_insights_report.txt
    If these files exist, they are included in the final HTML. If not, you’ll see “NOT FOUND.”
    """
    try:
        if not os.path.exists(REPORTS_DIR):
            raise HTTPException(status_code=404, detail="Reports directory not found.")

        # --- EDA file ---
        eda_prefix = f"eda_report_{dataset_id}"  # e.g. "eda_report_1234.txt"
        eda_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith(eda_prefix)]
        if eda_files:
            eda_content = ""
            for file in eda_files:
                file_path = os.path.join(REPORTS_DIR, file)
                logging.info(f"Loading EDA file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                # wrap EDA in <pre> (or parse further if you want)
                eda_content += f"<pre>{text}</pre><br/>"
        else:
            eda_content = "<p>EDA report: NOT FOUND</p>"

        # --- Training file ---
        train_prefix = f"training_results_{dataset_id}"
        train_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith(train_prefix)]
        if train_files:
            training_content = ""
            for file in train_files:
                file_path = os.path.join(REPORTS_DIR, file)
                logging.info(f"Loading Training file: {file_path}")

                if file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    training_content += "<h3>Training Results (CSV)</h3>"
                    training_content += df.to_html(index=False, justify='left')
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        txt = f.read()
                    training_content += f"<pre>{txt}</pre><br/>"
        else:
            training_content = "<p>Training report: NOT FOUND</p>"

        # --- Forecast log ---
        forecast_log_path = os.path.join(REPORTS_DIR, FORECAST_LOG_FILENAME)
        if os.path.exists(forecast_log_path):
            with open(forecast_log_path, "r", encoding="utf-8") as f:
                forecast_log_content = f"<pre>{f.read()}</pre>"
        else:
            forecast_log_content = "<p>Forecasting log: NOT FOUND</p>"

        # --- AI Insights ---
        ai_insights_path = os.path.join(REPORTS_DIR, AI_INSIGHTS_FILENAME)
        if os.path.exists(ai_insights_path):
            with open(ai_insights_path, "r", encoding="utf-8") as f:
                ai_insights_content = f"<pre>{f.read()}</pre>"
        else:
            ai_insights_content = "<p>AI insights: NOT FOUND</p>"

        # --- RAG Summary placeholder (if you have a file, load it, else placeholder)
        rag_summary_content = "<p>RAG summary: Not provided. Please run a RAG query if needed.</p>"

        # --- Optional: Executive Summary
        executive_summary = """
        <ul>
          <li><strong>Data Size:</strong> [Add dynamic info or leave as a placeholder]</li>
          <li><strong>Best Model Performance:</strong> [Add R^2 or Accuracy details]</li>
          <li><strong>Major Findings:</strong> [Brief bullet points about your data]</li>
        </ul>
        """

        # --- Final HTML ---
        full_report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Comprehensive Analysis Report</title>
            <style>
                body {{
                    font-family: "Open Sans", Arial, sans-serif;
                    background-color: #f9f9f9;
                    margin: 0; padding: 20px; color: #333;
                }}
                h1 {{
                    text-align: center; color: #2c3e50; margin-bottom: 10px;
                }}
                h2 {{
                    color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 5px;
                }}
                h3 {{
                    color: #2c3e50; margin-top: 20px;
                }}
                .section {{
                    background: #fff; margin: 20px 0; padding: 20px;
                    border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                pre {{
                    background: #ecf0f1; padding: 10px; border-radius: 4px;
                    overflow-x: auto; white-space: pre-wrap;
                }}
                table {{
                    border-collapse: collapse; width: 100%; margin: 10px 0;
                }}
                table, th, td {{
                    border: 1px solid #ccc;
                }}
                th, td {{
                    padding: 8px; text-align: left;
                }}
                .executive-summary {{
                    background-color: #fff;
                    padding: 10px; margin: 20px 0; border-radius: 6px;
                }}
                .footer {{
                    text-align: center; font-size: 0.9em; color: #7f8c8d; margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Comprehensive Analysis Report</h1>

            <div class="executive-summary">
                <h2>Executive Summary</h2>
                {executive_summary}
            </div>

            <div class="section">
                <h2>EDA Report</h2>
                {eda_content}
            </div>

            <div class="section">
                <h2>Training Report</h2>
                {training_content}
            </div>

            <div class="section">
                <h2>Forecasting Log</h2>
                {forecast_log_content}
            </div>

            <div class="section">
                <h2>AI Insights</h2>
                {ai_insights_content}
            </div>

            <div class="section">
                <h2>RAG Summary</h2>
                {rag_summary_content}
            </div>

            <div class="footer">
                <p>Report generated by AI Auto Dashboard</p>
            </div>
        </body>
        </html>
        """

        full_report_path = os.path.join(REPORTS_DIR, FULL_REPORT_FILENAME)
        with open(full_report_path, "w", encoding="utf-8") as f:
            f.write(full_report_content)

        logging.info(f"Full report generated at {os.path.abspath(full_report_path)}")
        return {"status": "success", "message": "Full report generated successfully."}

    except Exception as e:
        logging.error(f"Failed to generate full report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/view-full")
async def view_full_report():
    """
    Return the full HTML report inline for preview.
    """
    full_report_path = os.path.join(REPORTS_DIR, FULL_REPORT_FILENAME)
    if not os.path.exists(full_report_path):
        raise HTTPException(status_code=404, detail="Full report not found. Please generate it first.")
    return FileResponse(
        os.path.abspath(full_report_path),
        media_type="text/html",
        filename="full_report.html",
        headers={"Content-Disposition": "inline"}
    )


@router.get("/download-full")
async def download_full_report():
    """
    Convert the generated HTML report to PDF using Chrome headless mode, then return it.
    """
    full_report_path = os.path.join(REPORTS_DIR, FULL_REPORT_FILENAME)
    if not os.path.exists(full_report_path):
        raise HTTPException(status_code=404, detail="Full report not found. Please generate it first.")

    try:
        # Path to your Chrome or Chrome-based browser
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

        pdf_output_path = os.path.abspath(os.path.join(REPORTS_DIR, FULL_REPORT_PDF))
        os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)

        command = [
            chrome_path,
            '--headless',
            '--disable-gpu',
            f'--print-to-pdf={pdf_output_path}',
            os.path.abspath(full_report_path)
        ]
        logging.info(f"Running Chrome command: {' '.join(command)}")
        subprocess.run(command, check=True)

        if not os.path.exists(pdf_output_path):
            raise FileNotFoundError(f"PDF file not created at {pdf_output_path}.")

        return FileResponse(pdf_output_path, media_type="application/pdf", filename="full_report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert report to PDF: {str(e)}")
