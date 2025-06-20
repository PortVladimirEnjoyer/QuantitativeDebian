from conditionnal_var import main as conditional_var_main
from garch import main_garch

def risk():
    while True:
        # Simplified header for better compatibility
        print("\n=== RISK MODELING TOOL ===")
        print("1: Conditional Var")
        print("2: GARCH")
        print("3: Exit")

        try:
            choice = int(input("Select an option between 1 and 3: "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3.")
            continue

        if choice == 1:
            conditional_var_main()  # Actually call the function
        elif choice == 2:
            main_garch()  # Actually call the function
        elif choice == 3:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    risk()
