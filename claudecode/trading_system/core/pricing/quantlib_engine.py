"""
QuantLib-powered options pricing engine for professional derivatives pricing.
"""

import QuantLib as ql
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from enum import Enum
from loguru import logger

from ..base.exceptions import PricingError, ValidationError


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class PricingModel(Enum):
    """Available pricing models."""
    BLACK_SCHOLES = "black_scholes"
    HESTON = "heston"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"
    FINITE_DIFFERENCE = "finite_difference"


class QuantLibPricingEngine:
    """
    Professional-grade options pricing engine using QuantLib.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="quantlib_pricing")
        
        # Set global evaluation date
        self.evaluation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self.evaluation_date
        
        # Initialize calendars and day counters
        self.calendar = ql.UnitedStates()
        self.day_counter = ql.Actual365Fixed()
        
        self.logger.info("QuantLib pricing engine initialized")
    
    def price_vanilla_option(self, 
                           spot_price: float,
                           strike: float,
                           risk_free_rate: float,
                           dividend_yield: float,
                           volatility: float,
                           maturity_date: datetime,
                           option_type: OptionType,
                           model: PricingModel = PricingModel.BLACK_SCHOLES) -> Dict[str, Any]:
        """
        Price a vanilla European option using specified model.
        
        Args:
            spot_price: Current underlying price
            strike: Strike price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            volatility: Implied volatility
            maturity_date: Option expiration date
            option_type: Call or Put
            model: Pricing model to use
            
        Returns:
            Dictionary with price and Greeks
        """
        try:
            # Convert datetime to QuantLib date
            maturity_ql = self._datetime_to_ql_date(maturity_date)
            
            # Create option payoff and exercise
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put,
                strike
            )
            exercise = ql.EuropeanExercise(maturity_ql)
            option = ql.VanillaOption(payoff, exercise)
            
            # Set up market data
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
            )
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, dividend_yield, self.day_counter)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.evaluation_date, self.calendar, volatility, self.day_counter)
            )
            
            # Create BSM process
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, dividend_ts, flat_ts, flat_vol_ts
            )
            
            # Choose pricing engine based on model
            if model == PricingModel.BLACK_SCHOLES:
                engine = ql.AnalyticEuropeanEngine(bsm_process)
            elif model == PricingModel.BINOMIAL:
                engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
            elif model == PricingModel.MONTE_CARLO:
                engine = ql.MCEuropeanEngine(bsm_process, "PseudoRandom", timeSteps=1, requiredSamples=100000)
            elif model == PricingModel.FINITE_DIFFERENCE:
                engine = ql.FdBlackScholesVanillaEngine(bsm_process, 100, 100)
            else:
                engine = ql.AnalyticEuropeanEngine(bsm_process)
            
            option.setPricingEngine(engine)
            
            # Calculate price and Greeks
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
            vega = option.vega()
            rho = option.rho()
            
            results = {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Convert to daily theta
                'vega': vega / 100,    # Convert to vega per 1% vol change
                'rho': rho / 100,      # Convert to rho per 1% rate change
                'underlying_price': spot_price,
                'strike': strike,
                'time_to_expiry': (maturity_date - datetime.now()).days / 365.0,
                'volatility': volatility,
                'model_used': model.value
            }
            
            self.logger.debug(f"Priced {option_type.value} option: ${price:.4f}")
            return results
            
        except Exception as e:
            raise PricingError(f"Vanilla option pricing failed: {str(e)}")
    
    def price_american_option(self,
                            spot_price: float,
                            strike: float,
                            risk_free_rate: float,
                            dividend_yield: float,
                            volatility: float,
                            maturity_date: datetime,
                            option_type: OptionType,
                            steps: int = 100) -> Dict[str, Any]:
        """
        Price American-style option using binomial model.
        """
        try:
            maturity_ql = self._datetime_to_ql_date(maturity_date)
            
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put,
                strike
            )
            exercise = ql.AmericanExercise(self.evaluation_date, maturity_ql)
            option = ql.VanillaOption(payoff, exercise)
            
            # Market data setup
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
            )
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, dividend_yield, self.day_counter)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.evaluation_date, self.calendar, volatility, self.day_counter)
            )
            
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, dividend_ts, flat_ts, flat_vol_ts
            )
            
            # Use binomial engine for American options
            engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
            option.setPricingEngine(engine)
            
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
            vega = option.vega()
            rho = option.rho()
            
            results = {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,
                'vega': vega / 100,
                'rho': rho / 100,
                'exercise_style': 'american',
                'model_used': 'binomial'
            }
            
            return results
            
        except Exception as e:
            raise PricingError(f"American option pricing failed: {str(e)}")
    
    def calibrate_heston_model(self,
                             spot_price: float,
                             risk_free_rate: float,
                             dividend_yield: float,
                             market_prices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate Heston stochastic volatility model to market prices.
        
        Args:
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            market_prices: List of market option prices with strikes and maturities
            
        Returns:
            Calibrated Heston model parameters
        """
        try:
            # Set up market data
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
            )
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, dividend_yield, self.day_counter)
            )
            
            # Initial Heston parameters
            v0 = 0.04        # Initial variance
            kappa = 2.0      # Mean reversion speed
            theta = 0.04     # Long-term variance
            sigma = 0.3      # Volatility of volatility
            rho = -0.5       # Correlation
            
            heston_process = ql.HestonProcess(
                flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho
            )
            
            # Create calibration helpers
            helpers = []
            for market_data in market_prices:
                maturity_ql = self._datetime_to_ql_date(market_data['maturity'])
                
                helper = ql.HestonModelHelper(
                    ql.Period(int((market_data['maturity'] - datetime.now()).days), ql.Days),
                    self.calendar,
                    spot_price,
                    market_data['strike'],
                    ql.QuoteHandle(ql.SimpleQuote(market_data['market_price'])),
                    flat_ts,
                    dividend_ts
                )
                helpers.append(helper)
            
            # Create Heston model
            heston_model = ql.HestonModel(heston_process)
            engine = ql.AnalyticHestonEngine(heston_model)
            
            # Set pricing engine for helpers
            for helper in helpers:
                helper.setPricingEngine(engine)
            
            # Calibration
            optimization_method = ql.LevenbergMarquardt()
            heston_model.calibrate(helpers, optimization_method, ql.EndCriteria(1000, 50, 1.0e-8, 1.0e-8, 1.0e-8))
            
            # Extract calibrated parameters
            calibrated_params = {
                'v0': heston_model.v0(),
                'kappa': heston_model.kappa(),
                'theta': heston_model.theta(),
                'sigma': heston_model.sigma(),
                'rho': heston_model.rho(),
                'calibration_error': sum([helper.calibrationError() for helper in helpers])
            }
            
            self.logger.info(f"Heston model calibrated with error: {calibrated_params['calibration_error']:.6f}")
            return calibrated_params
            
        except Exception as e:
            raise PricingError(f"Heston calibration failed: {str(e)}")
    
    def price_barrier_option(self,
                           spot_price: float,
                           strike: float,
                           barrier: float,
                           risk_free_rate: float,
                           dividend_yield: float,
                           volatility: float,
                           maturity_date: datetime,
                           option_type: OptionType,
                           barrier_type: str = "down-and-out") -> Dict[str, Any]:
        """
        Price barrier options (knock-in/knock-out).
        """
        try:
            maturity_ql = self._datetime_to_ql_date(maturity_date)
            
            # Determine barrier type
            if barrier_type == "down-and-out":
                barrier_option_type = ql.Barrier.DownOut
            elif barrier_type == "up-and-out":
                barrier_option_type = ql.Barrier.UpOut
            elif barrier_type == "down-and-in":
                barrier_option_type = ql.Barrier.DownIn
            elif barrier_type == "up-and-in":
                barrier_option_type = ql.Barrier.UpIn
            else:
                raise ValidationError(f"Unknown barrier type: {barrier_type}")
            
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put,
                strike
            )
            exercise = ql.EuropeanExercise(maturity_ql)
            barrier_option = ql.BarrierOption(barrier_option_type, barrier, 0.0, payoff, exercise)
            
            # Market data
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
            )
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, dividend_yield, self.day_counter)
            )
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.evaluation_date, self.calendar, volatility, self.day_counter)
            )
            
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, dividend_ts, flat_ts, flat_vol_ts
            )
            
            # Analytic barrier option engine
            engine = ql.AnalyticBarrierEngine(bsm_process)
            barrier_option.setPricingEngine(engine)
            
            price = barrier_option.NPV()
            delta = barrier_option.delta()
            gamma = barrier_option.gamma()
            theta = barrier_option.theta()
            vega = barrier_option.vega()
            rho = barrier_option.rho()
            
            results = {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,
                'vega': vega / 100,
                'rho': rho / 100,
                'barrier_level': barrier,
                'barrier_type': barrier_type,
                'option_type': 'barrier'
            }
            
            return results
            
        except Exception as e:
            raise PricingError(f"Barrier option pricing failed: {str(e)}")
    
    def build_volatility_surface(self,
                                spot_price: float,
                                risk_free_rate: float,
                                dividend_yield: float,
                                market_data: List[Dict[str, Any]]) -> Any:
        """
        Build implied volatility surface from market option prices.
        
        Args:
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            market_data: List of option market data with strikes, maturities, and prices
            
        Returns:
            QuantLib BlackVarianceSurface object
        """
        try:
            # Organize data by maturity and strike
            maturities = sorted(list(set([data['maturity'] for data in market_data])))
            strikes = sorted(list(set([data['strike'] for data in market_data])))
            
            # Convert to QuantLib dates
            maturity_dates = [self._datetime_to_ql_date(mat) for mat in maturities]
            
            # Create volatility matrix
            vol_matrix = []
            for maturity in maturities:
                vol_row = []
                for strike in strikes:
                    # Find matching market data
                    matching_data = [d for d in market_data 
                                   if d['maturity'] == maturity and d['strike'] == strike]
                    
                    if matching_data:
                        # Solve for implied volatility
                        implied_vol = self._solve_implied_volatility(
                            spot_price, strike, risk_free_rate, dividend_yield,
                            matching_data[0]['market_price'], maturity, 
                            OptionType.CALL  # Assume call for surface building
                        )
                        vol_row.append(implied_vol)
                    else:
                        # Interpolate or use reasonable default
                        vol_row.append(0.20)  # 20% default volatility
                
                vol_matrix.append(vol_row)
            
            # Create QuantLib volatility surface
            vol_surface = ql.BlackVarianceSurface(
                self.evaluation_date,
                self.calendar,
                maturity_dates,
                strikes,
                vol_matrix,
                self.day_counter
            )
            
            return vol_surface
            
        except Exception as e:
            raise PricingError(f"Volatility surface construction failed: {str(e)}")
    
    def _solve_implied_volatility(self,
                                spot_price: float,
                                strike: float,
                                risk_free_rate: float,
                                dividend_yield: float,
                                market_price: float,
                                maturity_date: datetime,
                                option_type: OptionType) -> float:
        """Solve for implied volatility given market price."""
        try:
            maturity_ql = self._datetime_to_ql_date(maturity_date)
            
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put,
                strike
            )
            exercise = ql.EuropeanExercise(maturity_ql)
            option = ql.VanillaOption(payoff, exercise)
            
            # Market data
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, risk_free_rate, self.day_counter)
            )
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(self.evaluation_date, dividend_yield, self.day_counter)
            )
            
            # Solve for implied volatility
            implied_vol = ql.blackImpliedVol(
                market_price,
                spot_handle.value(),
                strike,
                flat_ts.forwardRate(self.evaluation_date, maturity_ql, self.day_counter, ql.Continuous).rate(),
                dividend_ts.forwardRate(self.evaluation_date, maturity_ql, self.day_counter, ql.Continuous).rate(),
                self.day_counter.yearFraction(self.evaluation_date, maturity_ql)
            )
            
            return implied_vol
            
        except Exception as e:
            self.logger.warning(f"Implied volatility calculation failed: {str(e)}")
            return 0.20  # Default fallback
    
    def _datetime_to_ql_date(self, dt: datetime) -> ql.Date:
        """Convert Python datetime to QuantLib Date."""
        return ql.Date(dt.day, dt.month, dt.year)
    
    def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks for multiple option positions.
        
        Args:
            positions: List of option positions with pricing data
            
        Returns:
            Dictionary with aggregated Greeks
        """
        try:
            portfolio_greeks = {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0,
                'total_value': 0.0
            }
            
            for position in positions:
                quantity = position.get('quantity', 0)
                pricing_data = position.get('pricing_data', {})
                
                portfolio_greeks['delta'] += quantity * pricing_data.get('delta', 0)
                portfolio_greeks['gamma'] += quantity * pricing_data.get('gamma', 0)
                portfolio_greeks['theta'] += quantity * pricing_data.get('theta', 0)
                portfolio_greeks['vega'] += quantity * pricing_data.get('vega', 0)
                portfolio_greeks['rho'] += quantity * pricing_data.get('rho', 0)
                portfolio_greeks['total_value'] += quantity * pricing_data.get('price', 0)
            
            return portfolio_greeks
            
        except Exception as e:
            raise PricingError(f"Portfolio Greeks calculation failed: {str(e)}")


class OptionsStrategyPricer:
    """
    Specialized pricer for complex options strategies.
    """
    
    def __init__(self, pricing_engine: QuantLibPricingEngine):
        self.pricing_engine = pricing_engine
        self.logger = logger.bind(component="strategy_pricer")
    
    def price_iron_condor(self,
                         spot_price: float,
                         strikes: Tuple[float, float, float, float],  # put_low, put_high, call_low, call_high
                         risk_free_rate: float,
                         dividend_yield: float,
                         volatility: float,
                         maturity_date: datetime) -> Dict[str, Any]:
        """Price iron condor strategy."""
        try:
            put_low_strike, put_high_strike, call_low_strike, call_high_strike = strikes
            
            # Price individual legs
            short_put = self.pricing_engine.price_vanilla_option(
                spot_price, put_high_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, OptionType.PUT
            )
            long_put = self.pricing_engine.price_vanilla_option(
                spot_price, put_low_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, OptionType.PUT
            )
            short_call = self.pricing_engine.price_vanilla_option(
                spot_price, call_low_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, OptionType.CALL
            )
            long_call = self.pricing_engine.price_vanilla_option(
                spot_price, call_high_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, OptionType.CALL
            )
            
            # Calculate strategy price and Greeks
            strategy_price = (short_put['price'] - long_put['price'] + 
                            short_call['price'] - long_call['price'])
            
            strategy_greeks = {
                'delta': (short_put['delta'] - long_put['delta'] + 
                         short_call['delta'] - long_call['delta']),
                'gamma': (short_put['gamma'] - long_put['gamma'] + 
                         short_call['gamma'] - long_call['gamma']),
                'theta': (short_put['theta'] - long_put['theta'] + 
                         short_call['theta'] - long_call['theta']),
                'vega': (short_put['vega'] - long_put['vega'] + 
                        short_call['vega'] - long_call['vega']),
                'rho': (short_put['rho'] - long_put['rho'] + 
                       short_call['rho'] - long_call['rho'])
            }
            
            # Calculate max profit/loss
            put_spread_width = put_high_strike - put_low_strike
            call_spread_width = call_high_strike - call_low_strike
            max_profit = strategy_price
            max_loss = min(put_spread_width, call_spread_width) - max_profit
            
            return {
                'strategy_price': strategy_price,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven_lower': put_high_strike - max_profit,
                'breakeven_upper': call_low_strike + max_profit,
                'greeks': strategy_greeks,
                'individual_legs': {
                    'short_put': short_put,
                    'long_put': long_put,
                    'short_call': short_call,
                    'long_call': long_call
                }
            }
            
        except Exception as e:
            raise PricingError(f"Iron condor pricing failed: {str(e)}")
    
    def price_butterfly_spread(self,
                             spot_price: float,
                             center_strike: float,
                             wing_width: float,
                             risk_free_rate: float,
                             dividend_yield: float,
                             volatility: float,
                             maturity_date: datetime,
                             option_type: OptionType) -> Dict[str, Any]:
        """Price butterfly spread strategy."""
        try:
            lower_strike = center_strike - wing_width
            upper_strike = center_strike + wing_width
            
            # Price individual legs
            long_lower = self.pricing_engine.price_vanilla_option(
                spot_price, lower_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, option_type
            )
            short_center = self.pricing_engine.price_vanilla_option(
                spot_price, center_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, option_type
            )
            long_upper = self.pricing_engine.price_vanilla_option(
                spot_price, upper_strike, risk_free_rate, dividend_yield,
                volatility, maturity_date, option_type
            )
            
            # Strategy price (buy 1, sell 2, buy 1)
            strategy_price = (long_lower['price'] - 2 * short_center['price'] + 
                            long_upper['price'])
            
            strategy_greeks = {
                'delta': (long_lower['delta'] - 2 * short_center['delta'] + 
                         long_upper['delta']),
                'gamma': (long_lower['gamma'] - 2 * short_center['gamma'] + 
                         long_upper['gamma']),
                'theta': (long_lower['theta'] - 2 * short_center['theta'] + 
                         long_upper['theta']),
                'vega': (long_lower['vega'] - 2 * short_center['vega'] + 
                        long_upper['vega']),
                'rho': (long_lower['rho'] - 2 * short_center['rho'] + 
                       long_upper['rho'])
            }
            
            max_profit = wing_width - abs(strategy_price)
            max_loss = abs(strategy_price)
            
            return {
                'strategy_price': strategy_price,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven_lower': center_strike - (wing_width - abs(strategy_price)),
                'breakeven_upper': center_strike + (wing_width - abs(strategy_price)),
                'greeks': strategy_greeks,
                'individual_legs': {
                    'long_lower': long_lower,
                    'short_center': short_center,
                    'long_upper': long_upper
                }
            }
            
        except Exception as e:
            raise PricingError(f"Butterfly spread pricing failed: {str(e)}")