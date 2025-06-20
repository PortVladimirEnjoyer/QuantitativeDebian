from gmb import monte_carlo
from lstm import main 
from random_forest import main 





def generative():
    while True:
        print("""
              


        [
                                                                       
 _____                     _   _         _____         _         _     
|   __|___ ___ ___ ___ ___| |_|_|_ _ ___|  _  |___ ___| |_ _ ___|_|___ 
|  |  | -_|   | -_|  _| .'|  _| | | | -_|     |   | .'| | | |_ -| |_ -|
|_____|___|_|_|___|_| |__,|_| |_|\_/|___|__|__|_|_|__,|_|_  |___|_|___|
                                                        |___|          

              


        """
              
              )


        print("1: Monte Carlo")
        print("2: Lstm")
        print("3: random forest")
        print("4: exit")

        try:
            choice = int(input("Select an option between 1 and 5: "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue

        if choice == 1:
            monte_carlo()
        elif choice == 2:
            main() #lstm 
        elif choice == 3:
            main() #random forest 

        elif choice == 4:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    generative()


