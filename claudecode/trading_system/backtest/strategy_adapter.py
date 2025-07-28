"""
Strategy adapter that connects our trading agents to Backtrader.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
import asyncio

from ..core.base.exceptions import BacktestError


class StrategyAdapter:
    """
    Adapter that converts our agent-based strategies into Backtrader-compatible format.
    """
    
    def __init__(self, strategy_config: Dict[str, Any]):
        self.config = strategy_config
        self.strategy_type = strategy_config.get('type', 'momentum')
        self.agent_config = strategy_config.get('agent_config', {})
        self.logger = logger.bind(strategy_adapter=self.strategy_type)
        
        # Initialize strategy-specific components
        self._initialize_strategy_components()
    
    def _initialize_strategy_components(self):
        """Initialize components based on strategy type."""
        try:
            if self.strategy_type == 'momentum':
                self._init_momentum_strategy()
            elif self.strategy_type == 'stat_arb':
                self._init_stat_arb_strategy()
            elif self.strategy_type == 'event_driven':
                self._init_event_driven_strategy()
            elif self.strategy_type == 'options':
                self._init_options_strategy()
            elif self.strategy_type == 'cross_asset':
                self._init_cross_asset_strategy()
            elif self.strategy_type == 'multi_agent':
                self._init_multi_agent_strategy()
            else:
                raise BacktestError(f"Unknown strategy type: {self.strategy_type}")
                
            self.logger.info(f"Initialized {self.strategy_type} strategy adapter")
            
        except Exception as e:
            raise BacktestError(f"Strategy initialization failed: {str(e)}")
    
    def _init_momentum_strategy(self):
        """Initialize momentum strategy components."""
        self.lookback_period = self.agent_config.get('lookback_period', 20)
        self.momentum_threshold = self.agent_config.get('threshold', 0.02)
        self.min_volume = self.agent_config.get('min_volume', 1000000)
        self.max_positions = self.agent_config.get('max_positions', 10)
        
        # Momentum calculation parameters
        self.momentum_windows = self.agent_config.get('momentum_windows', [5, 10, 20])
        self.volume_confirmation = self.agent_config.get('volume_confirmation', True)
    
    def _init_stat_arb_strategy(self):
        """Initialize statistical arbitrage strategy components."""
        self.pairs_method = self.agent_config.get('pairs_method', 'cointegration')
        self.z_score_entry = self.agent_config.get('z_score_entry', 2.0)
        self.z_score_exit = self.agent_config.get('z_score_exit', 0.5)
        self.lookback_period = self.agent_config.get('lookback_period', 252)
        self.max_pairs = self.agent_config.get('max_pairs', 5)
    
    def _init_event_driven_strategy(self):
        """Initialize event-driven strategy components."""
        self.sentiment_threshold = self.agent_config.get('sentiment_threshold', 0.6)
        self.event_window = self.agent_config.get('event_window', 5)
        self.news_sources = self.agent_config.get('news_sources', ['general'])
        self.earnings_weight = self.agent_config.get('earnings_weight', 0.4)
    
    def _init_options_strategy(self):
        """Initialize options strategy components."""
        self.strategies = self.agent_config.get('strategies', ['iron_condor'])
        self.max_dte = self.agent_config.get('max_dte', 45)
        self.min_prob_profit = self.agent_config.get('min_prob_profit', 0.6)
        self.volatility_threshold = self.agent_config.get('volatility_threshold', 0.2)
    
    def _init_cross_asset_strategy(self):
        """Initialize cross-asset strategy components."""
        self.correlation_threshold = self.agent_config.get('correlation_threshold', 0.7)
        self.regime_detection = self.agent_config.get('regime_detection', True)
        self.rebalance_frequency = self.agent_config.get('rebalance_frequency', 'monthly')
        self.diversification_factor = self.agent_config.get('diversification_factor', 0.8)
    
    def _init_multi_agent_strategy(self):
        """Initialize multi-agent ensemble strategy."""
        self.agent_weights = self.agent_config.get('agent_weights', {
            'momentum': 0.3,
            'stat_arb': 0.2,
            'event_driven': 0.2,
            'cross_asset': 0.3
        })
        self.consensus_threshold = self.agent_config.get('consensus_threshold', 0.6)
        self.min_confirming_agents = self.agent_config.get('min_confirming_agents', 2)
    
    def get_recommendations(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """
        Generate trading recommendations based on strategy type and market data.
        
        Args:
            market_data: Current market data for all symbols
            current_date: Current date in backtest
            
        Returns:
            List of trading recommendations
        """
        try:
            if self.strategy_type == 'momentum':
                return self._generate_momentum_signals(market_data, current_date)
            elif self.strategy_type == 'stat_arb':
                return self._generate_stat_arb_signals(market_data, current_date)
            elif self.strategy_type == 'event_driven':
                return self._generate_event_driven_signals(market_data, current_date)
            elif self.strategy_type == 'options':
                return self._generate_options_signals(market_data, current_date)
            elif self.strategy_type == 'cross_asset':
                return self._generate_cross_asset_signals(market_data, current_date)
            elif self.strategy_type == 'multi_agent':
                return self._generate_multi_agent_signals(market_data, current_date)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return []
    
    def _generate_momentum_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate momentum-based trading signals."""
        try:
            recommendations = []
            
            for symbol, data in market_data.items():
                # Calculate momentum score (simplified)
                current_price = data.get('close', 0)
                volume = data.get('volume', 0)
                
                if current_price <= 0 or volume < self.min_volume:
                    continue
                
                # Simplified momentum calculation
                # In real implementation, this would use historical price data
                momentum_score = np.random.normal(0, 0.1)  # Placeholder
                
                if abs(momentum_score) > self.momentum_threshold:
                    action = 'BUY' if momentum_score > 0 else 'SELL'
                    confidence = min(abs(momentum_score) / self.momentum_threshold, 1.0)
                    position_size = confidence * 0.1  # Max 10% position
                    
                    recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'position_size': position_size,
                        'confidence': confidence,
                        'strategy': 'momentum',
                        'signal_strength': abs(momentum_score),
                        'timestamp': current_date
                    })
            
            # Limit number of positions
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            return recommendations[:self.max_positions]
            
        except Exception as e:
            self.logger.error(f"Momentum signal generation failed: {str(e)}")
            return []
    
    def _generate_stat_arb_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate statistical arbitrage signals."""
        try:
            recommendations = []
            symbols = list(market_data.keys())
            
            # Generate pairs signals (simplified)
            for i in range(0, len(symbols)-1, 2):
                if i+1 >= len(symbols):
                    break
                
                symbol1, symbol2 = symbols[i], symbols[i+1]
                data1, data2 = market_data[symbol1], market_data[symbol2]
                
                price1, price2 = data1.get('close', 0), data2.get('close', 0)
                
                if price1 <= 0 or price2 <= 0:
                    continue
                
                # Simplified z-score calculation (placeholder)
                z_score = np.random.normal(0, 1.5)  # Placeholder
                
                if abs(z_score) > self.z_score_entry:
                    # Enter pair trade
                    if z_score > 0:  # Symbol1 overvalued, Symbol2 undervalued
                        recommendations.extend([
                            {
                                'symbol': symbol1,
                                'action': 'SELL',
                                'position_size': 0.05,
                                'confidence': min(abs(z_score) / self.z_score_entry, 1.0),
                                'strategy': 'stat_arb',
                                'pair_symbol': symbol2,
                                'z_score': z_score,
                                'timestamp': current_date
                            },
                            {
                                'symbol': symbol2,
                                'action': 'BUY',
                                'position_size': 0.05,
                                'confidence': min(abs(z_score) / self.z_score_entry, 1.0),
                                'strategy': 'stat_arb',
                                'pair_symbol': symbol1,
                                'z_score': -z_score,
                                'timestamp': current_date
                            }
                        ])
            
            return recommendations[:self.max_pairs * 2]
            
        except Exception as e:
            self.logger.error(f"Stat arb signal generation failed: {str(e)}")
            return []
    
    def _generate_event_driven_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate event-driven signals."""
        try:
            recommendations = []
            
            for symbol, data in market_data.items():
                # Simulate event detection (placeholder)
                has_event = np.random.random() < 0.1  # 10% chance of event
                
                if has_event:
                    sentiment_score = np.random.uniform(-1, 1)  # Placeholder sentiment
                    
                    if abs(sentiment_score) > self.sentiment_threshold:
                        action = 'BUY' if sentiment_score > 0 else 'SELL'
                        confidence = abs(sentiment_score)
                        position_size = confidence * 0.08  # Max 8% position
                        
                        recommendations.append({
                            'symbol': symbol,
                            'action': action,
                            'position_size': position_size,
                            'confidence': confidence,
                            'strategy': 'event_driven',
                            'sentiment_score': sentiment_score,
                            'event_type': 'news',  # Placeholder
                            'timestamp': current_date
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Event-driven signal generation failed: {str(e)}")
            return []
    
    def _generate_options_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate options strategy signals."""
        try:
            recommendations = []
            
            for symbol, data in market_data.items():
                # Simplified options signal (placeholder)
                current_price = data.get('close', 0)
                
                if current_price <= 0:
                    continue
                
                # Simulate volatility and options metrics
                implied_vol = np.random.uniform(0.15, 0.35)  # Placeholder
                
                if implied_vol > self.volatility_threshold:
                    # High volatility - sell premium
                    recommendations.append({
                        'symbol': symbol,
                        'action': 'SELL_OPTION',
                        'position_size': 0.03,  # Small options position
                        'confidence': 0.7,
                        'strategy': 'options',
                        'option_strategy': 'iron_condor',
                        'implied_vol': implied_vol,
                        'timestamp': current_date
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Options signal generation failed: {str(e)}")
            return []
    
    def _generate_cross_asset_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate cross-asset allocation signals."""
        try:
            recommendations = []
            symbols = list(market_data.keys())
            
            # Simplified asset allocation (equal weight with small random adjustments)
            base_weight = 1.0 / len(symbols) if symbols else 0
            
            for symbol in symbols:
                # Add small random variation to weights
                weight_adjustment = np.random.uniform(-0.02, 0.02)
                target_weight = max(0.01, base_weight + weight_adjustment)
                
                recommendations.append({
                    'symbol': symbol,
                    'action': 'REBALANCE',
                    'position_size': target_weight,
                    'confidence': 0.8,
                    'strategy': 'cross_asset',
                    'allocation_type': 'diversified',
                    'timestamp': current_date
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Cross-asset signal generation failed: {str(e)}")
            return []
    
    def _generate_multi_agent_signals(self, market_data: Dict[str, Any], current_date: datetime) -> List[Dict[str, Any]]:
        """Generate multi-agent ensemble signals."""
        try:
            # Get signals from each agent type
            all_signals = {}
            
            if 'momentum' in self.agent_weights:
                momentum_adapter = StrategyAdapter({'type': 'momentum', 'agent_config': {}})
                momentum_signals = momentum_adapter._generate_momentum_signals(market_data, current_date)
                all_signals['momentum'] = momentum_signals
            
            if 'stat_arb' in self.agent_weights:
                stat_arb_adapter = StrategyAdapter({'type': 'stat_arb', 'agent_config': {}})
                stat_arb_signals = stat_arb_adapter._generate_stat_arb_signals(market_data, current_date)
                all_signals['stat_arb'] = stat_arb_signals
            
            if 'event_driven' in self.agent_weights:
                event_adapter = StrategyAdapter({'type': 'event_driven', 'agent_config': {}})
                event_signals = event_adapter._generate_event_driven_signals(market_data, current_date)
                all_signals['event_driven'] = event_signals
            
            if 'cross_asset' in self.agent_weights:
                cross_adapter = StrategyAdapter({'type': 'cross_asset', 'agent_config': {}})
                cross_signals = cross_adapter._generate_cross_asset_signals(market_data, current_date)
                all_signals['cross_asset'] = cross_signals
            
            # Combine signals using weighted consensus
            consensus_signals = self._combine_agent_signals(all_signals)
            
            return consensus_signals
            
        except Exception as e:
            self.logger.error(f"Multi-agent signal generation failed: {str(e)}")
            return []
    
    def _combine_agent_signals(self, all_signals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine signals from multiple agents using weighted consensus."""
        try:
            symbol_signals = {}
            
            # Group signals by symbol
            for agent_type, signals in all_signals.items():
                weight = self.agent_weights.get(agent_type, 0)
                
                for signal in signals:
                    symbol = signal.get('symbol')
                    if not symbol:
                        continue
                    
                    if symbol not in symbol_signals:
                        symbol_signals[symbol] = []
                    
                    # Add weighted signal
                    weighted_signal = signal.copy()
                    weighted_signal['agent_weight'] = weight
                    weighted_signal['weighted_confidence'] = signal.get('confidence', 0) * weight
                    symbol_signals[symbol].append(weighted_signal)
            
            # Generate consensus signals
            consensus_recommendations = []
            
            for symbol, signals in symbol_signals.items():
                if len(signals) < self.min_confirming_agents:
                    continue
                
                # Calculate weighted consensus
                total_weighted_confidence = sum(s.get('weighted_confidence', 0) for s in signals)
                
                if total_weighted_confidence >= self.consensus_threshold:
                    # Determine consensus action
                    buy_weight = sum(s.get('weighted_confidence', 0) for s in signals if s.get('action') == 'BUY')
                    sell_weight = sum(s.get('weighted_confidence', 0) for s in signals if s.get('action') == 'SELL')
                    
                    if buy_weight > sell_weight:
                        action = 'BUY'
                        confidence = buy_weight
                    elif sell_weight > buy_weight:
                        action = 'SELL'
                        confidence = sell_weight
                    else:
                        action = 'HOLD'
                        confidence = total_weighted_confidence
                    
                    # Calculate consensus position size
                    avg_position_size = np.mean([s.get('position_size', 0) for s in signals])
                    
                    consensus_recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'position_size': avg_position_size,
                        'confidence': confidence,
                        'strategy': 'multi_agent',
                        'contributing_agents': [s.get('strategy') for s in signals],
                        'consensus_score': total_weighted_confidence,
                        'timestamp': signals[0].get('timestamp')
                    })
            
            return consensus_recommendations
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {str(e)}")
            return []