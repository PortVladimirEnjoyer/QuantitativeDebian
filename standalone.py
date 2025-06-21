#note : ai was used for thte conception of this code 


#!/usr/bin/env python3
"""

Uses subprocess to run all functions without modifications
"""
import argparse
import sys
import subprocess
from datetime import datetime
from main import main_main

def display_main_banner():
    print("""
_________________________________________________________
                   QUant
_________________________________________________________
""")

def validate_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {date_str}. Use YYYY-MM-DD")

def run_subprocess(script_name, input_sequence=""):
    """Generic subprocess runner"""
    try:
        subprocess.run(
            ["python", script_name],
            input=input_sequence,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

def run_lstm(args):
    """Run LSTM with parameters"""
    input_seq = f"{args.ticker}\n{args.start}\n{args.end}\n"
    run_subprocess("lstm.py", input_seq)

def run_cvar(args):
    """Run CVaR with parameters"""
    input_seq = f"{args.confidence}\n2\n{args.ticker}\n"
    run_subprocess("conditionnal_var.py", input_seq)

def run_garch(args):
    """Run GARCH with parameters"""
    input_seq = f"2\n{args.ticker}\n{args.model_type}\n{args.p}\n{args.q}\n{args.horizon}\n"
    run_subprocess("garch.py", input_seq)

def run_montecarlo(args):
    """Run Monte Carlo with parameters"""
    input_seq = (
        f"{args.ticker}\n{args.initial_price}\n{args.return_rate}\n"
        f"{args.volatility}\n{args.years}\n{args.simulations}\n"
    )
    run_subprocess("gmb.py", input_seq)

def run_randomforest(args):
    """Run Random Forest with parameters"""
    input_seq = f"{args.ticker}\n{args.start}\n{args.end}\n{args.days}\n"
    run_subprocess("random_forest.py", input_seq)

def main():
    parser = argparse.ArgumentParser(description='Financial Analysis Toolkit')
    subparsers = parser.add_subparsers(dest='command')
    
    # Generative commands
    gen_parser = subparsers.add_parser('generative', help='Generative models')
    gen_subparsers = gen_parser.add_subparsers(dest='model')
    
    # LSTM
    lstm_parser = gen_subparsers.add_parser('lstm', help='LSTM price prediction')
    lstm_parser.add_argument('--ticker', required=True, help='Stock symbol')
    lstm_parser.add_argument('--start', required=True, type=validate_date, help='Start date (YYYY-MM-DD)')
    lstm_parser.add_argument('--end', required=True, type=validate_date, help='End date (YYYY-MM-DD)')
    lstm_parser.set_defaults(func=run_lstm)
    
    # Monte Carlo
    mc_parser = gen_subparsers.add_parser('montecarlo', help='Monte Carlo simulation')
    mc_parser.add_argument('--ticker', required=True, help='Asset name')
    mc_parser.add_argument('--initial-price', type=float, required=True, help='Initial price')
    mc_parser.add_argument('--return-rate', type=float, required=True, help='Expected annual return')
    mc_parser.add_argument('--volatility', type=float, required=True, help='Annual volatility')
    mc_parser.add_argument('--years', type=float, required=True, help='Time horizon in years')
    mc_parser.add_argument('--simulations', type=int, required=True, help='Number of simulations')
    mc_parser.set_defaults(func=run_montecarlo)
    
    # Random Forest
    rf_parser = gen_subparsers.add_parser('randomforest', help='Random Forest prediction')
    rf_parser.add_argument('--ticker', required=True, help='Stock symbol')
    rf_parser.add_argument('--start', required=True, type=validate_date, help='Start date (YYYY-MM-DD)')
    rf_parser.add_argument('--end', required=True, type=validate_date, help='End date (YYYY-MM-DD)')
    rf_parser.add_argument('--days', type=int, required=True, help='Days to forecast (1-30)')
    rf_parser.set_defaults(func=run_randomforest)
    
    # Risk commands
    risk_parser = subparsers.add_parser('risk', help='Risk analysis tools')
    risk_subparsers = risk_parser.add_subparsers(dest='tool')
    
    # CVaR
    cvar_parser = risk_subparsers.add_parser('cvar', help='Conditional Value at Risk')
    cvar_parser.add_argument('--confidence', type=float, required=True, help='Confidence level (e.g. 95)')
    cvar_parser.add_argument('--ticker', required=True, help='Asset ticker')
    cvar_parser.set_defaults(func=run_cvar)
    
    # GARCH
    garch_parser = risk_subparsers.add_parser('garch', help='GARCH volatility model')
    garch_parser.add_argument('--ticker', required=True, help='Asset ticker')
    garch_parser.add_argument('--model-type', choices=['1','2','3'], required=True, 
                            help='1=GARCH, 2=EGARCH, 3=GJR-GARCH')
    garch_parser.add_argument('--p', type=int, required=True, help='GARCH order (p)')
    garch_parser.add_argument('--q', type=int, required=True, help='ARCH order (q)')
    garch_parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon (days)')
    garch_parser.set_defaults(func=run_garch)
    
    # LLM command
    subparsers.add_parser('llm', help='AI assistant').set_defaults(func=lambda _: run_subprocess("llm.py"))
    
    args = parser.parse_args()
    
    if not args.command:
        display_main_banner()
        main_main()
        return
    
    try:
        if hasattr(args, 'func'):
            args.func(args)
        elif args.command == 'generative':
            run_subprocess("main_generative.py")
        elif args.command == 'risk':
            run_subprocess("main_risk.py")
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(0)

if __name__ == "__main__":
    main()
