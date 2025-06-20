from main_generative import generative
from llm import main as main_llm
from main_risk import risk 


def main_main():
  while True:
    print( """
________ __                                          __            __        ________                                 __                      __
/        /  |                                        /  |          /  |      /        |                               /  |                    /  |
$$$$$$$$/$$/  _______    ______   _______    _______ $$/   ______  $$ |      $$$$$$$$/______    ______   _____  ____  $$/  _______    ______  $$ |
$$ |__   /  |/       \  /      \ /       \  /       |/  | /      \ $$ |         $$ | /      \  /      \ /     \/    \ /  |/       \  /      \ $$ |
$$    |  $$ |$$$$$$$  | $$$$$$  |$$$$$$$  |/$$$$$$$/ $$ | $$$$$$  |$$ |         $$ |/$$$$$$  |/$$$$$$  |$$$$$$ $$$$  |$$ |$$$$$$$  | $$$$$$  |$$ |
$$$$$/   $$ |$$ |  $$ | /    $$ |$$ |  $$ |$$ |      $$ | /    $$ |$$ |         $$ |$$    $$ |$$ |  $$/ $$ | $$ | $$ |$$ |$$ |  $$ | /    $$ |$$ |
$$ |     $$ |$$ |  $$ |/$$$$$$$ |$$ |  $$ |$$ \_____ $$ |/$$$$$$$ |$$ |         $$ |$$$$$$$$/ $$ |      $$ | $$ | $$ |$$ |$$ |  $$ |/$$$$$$$ |$$ |
$$ |     $$ |$$ |  $$ |$$    $$ |$$ |  $$ |$$       |$$ |$$    $$ |$$ |         $$ |$$       |$$ |      $$ | $$ | $$ |$$ |$$ |  $$ |$$    $$ |$$ |
$$/      $$/ $$/   $$/  $$$$$$$/ $$/   $$/  $$$$$$$/ $$/  $$$$$$$/ $$/          $$/  $$$$$$$/ $$/       $$/  $$/  $$/ $$/ $$/   $$/  $$$$$$$/ $$/

                                                                                                                                                  """


)
    print("...for generative analysis : press 1")
    print("...to start llm model (wizardMath): press 2")
    print("...for risk optimization : press 3")
    try:
      choice = int(input("..."))
    except ValueError:
      print("chose only a number between 1 and 3")
    if choice == 1:
      print("...Starting generative analysis")
      generative()
    if choice == 2:
      print("...start llm model ")
      main_llm()
    if choice == 3:
      print("...starting risk optimization")
      risk()

if __name__ == "__main__":
    main_main()
