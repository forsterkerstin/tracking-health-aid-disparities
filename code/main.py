import importlib.util
import os
import sys

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_plot_directories():
    # Create the 'plots' folder and its subfolders at the root level if they don't exist
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    subfolders = ["aid_plots", "classification", "correlation", "map_plots"]

    for folder in [plots_dir] + [
        os.path.join(plots_dir, subfolder) for subfolder in subfolders
    ]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
        else:
            print(f"Directory already exists: {folder}")


def create_data_directories():
    """
    Creates the 'data/results' folder at the project root level if it doesn't exist.
    This is necessary for scripts that output processed data or results.
    """
    data_results_dir = os.path.join(PROJECT_ROOT, "data", "results")
    if not os.path.exists(data_results_dir):
        os.makedirs(data_results_dir)
        print(f"Created directory: {data_results_dir}")
    else:
        print(f"Directory already exists: {data_results_dir}")


def run_script(script_name):
    # Construct the full path to the script
    script_path = os.path.join(
        PROJECT_ROOT, "code", "Visualization", script_name
    )

    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: {script_name} not found at {script_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of Visualization directory:")
        try:
            visualization_contents = os.listdir(os.path.dirname(script_path))
            for item in visualization_contents:
                print(f"  - {item}")
        except FileNotFoundError:
            print("  Visualization directory not found.")
        return

    # Add the script's directory to sys.path
    sys.path.insert(0, os.path.dirname(script_path))

    # Load the script as a module
    spec = importlib.util.spec_from_file_location(
        script_name[:-3], script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Run the main function if it exists
    if hasattr(module, "main"):
        module.main()
    else:
        print(f"Error: {script_name} does not have a main() function.")


def main():
    create_plot_directories()
    create_data_directories()

    print("run 07_aid_funding_plots.py")
    run_script("07_aid_funding_plots.py")
    print("07_aid_funding_plots.py done")

    print("run 08_correlation_and_funding_disparities_plots.py")
    run_script("08_correlation_and_funding_disparities_plots.py")
    print("08_correlation_and_funding_disparities_plots.py done")

    print("run 09_comparing_classification_methods.py")
    run_script("09_comparing_classification_methods.py")
    print("09_comparing_classification_methods.py done")


if __name__ == "__main__":
    main()
