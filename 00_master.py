import subprocess

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"Successfully processed {script_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to process {script_path}")
        return False

def run_notebook(notebook_path):
    try:
        subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            notebook_path
        ], check=True)
        print(f"Successfully processed {notebook_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to process {notebook_path}")
        return False

if __name__ == "__main__":
    workflow_steps = [
        ('script', '01_preprocess.py'),
        ('script', '02_feature_engineering.py'),
        ('script', '03_ml_model.py'),
        ('script', '04_dl_model.py')
    ]

    for step_type, file_name in workflow_steps:
        if step_type == 'script':
            success = run_script(file_name)
        elif step_type == 'notebook':
            success = run_notebook(file_name)
        
        if not success:
            print(f"Stopping workflow due to failure in {file_name}")
            break
