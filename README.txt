This project is a 3D physics-based simulator developed in Python
It is designed to conduct a comparative performance analysis of three guidance laws for drone interception:
PP, PN, and APN

This simulator was designed for Python 3.10
It requires the packages NumPy, Matplotlib, PyTest, Streamlit, and Plotly
Please run this command to install those packages:

pip install -r requirements.txt

---

There are three ways to run the simulation:

Option 1 - Web UI (recommended):
Run the following command from the project root:
    streamlit run src/app.py
Open the URL shown in your browser (usually http://localhost:8501)
Select a guidance law, target scenario, and seed from the sidebar, then click "Run Simulation"
An interactive 3D plot and metrics will be displayed

Option 2 - Single engagement (CLI):
Configure the scenario in src/config.py and set the GUIDANCE_LAW and TARGET_TEMPLATE_NAME variables
Target template list can be found in src/target_templates.py
Execute the script by running src/main.py from the terminal or IDE
An interactive 3D plot of the simulation run will be displayed

Option 3 - Batch run:
Configure the scenario in src/batch_run_main.py
Set TARGET_ORDER with a list of the targets you want to run and the order you want to run them in
Set BATCH_PLAN with the target templates and number of seeds for each
Execute the script by running src/batch_run_main.py from the terminal or IDE
The batch runner will create a folder named batch_out with PNGs and a compiled CSV file

---

To run unit tests:
    pytest unit_tests/unit_tests.py -v

- Peter Simpson, September 2025