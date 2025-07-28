"""
Main entry point for the Multi-Agent Trading System - Refactored Version.

This refactored version breaks down the complex run_analysis method into 
smaller, more manageable and testable methods to reduce cyclomatic complexity.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from loguru import logger
from core.utils.logging_setup import setup_logging
from core.base.config import config
from agents.data_universe import DataUniverseAgent
from agents.feature_engineering import TechnicalAnalysisAgent
from agents.ml_ensemble import MLEnsembleAgent
from agents.strategies.momentum import MomentumAgent
from agents.strategies.stat_arb import StatisticalArbitrageAgent
from agents.strategies.event_driven import EventDrivenAgent
from agents.strategies.options import OptionsAgent
from agents.strategies.cross_asset import CrossAssetAgent
from agents.synthesis import SignalSynthesisAgent
from agents.risk_management import RiskModelingAgent
from agents.output import RecommendationAgent


class TradingSystem:
    """
    Main trading system orchestrator that coordinates all agents.
    Refactored to reduce complexity and improve maintainability.
    """
    
    def __init__(self):
        self.agents = {
            'data_universe': DataUniverseAgent(),
            'technical_analysis': TechnicalAnalysisAgent(),
            'ml_ensemble': MLEnsembleAgent(),
            'momentum_strategy': MomentumAgent(),
            'stat_arb': StatisticalArbitrageAgent(),
            'event_driven': EventDrivenAgent(),
            'options_strategy': OptionsAgent(),
            'cross_asset': CrossAssetAgent(),
            'signal_synthesis': SignalSynthesisAgent(),
            'risk_modeling': RiskModelingAgent(),
            'recommendation': RecommendationAgent(),
        }
        
        setup_logging()
        self.logger = logger.bind(system="trading_system")
        
    async def run_analysis(self, 
                          start_date: datetime,
                          end_date: datetime,
                          asset_classes: list = None,
                          exchanges: list = None,
                          custom_symbols: list = None) -> Dict[str, Any]:
        """
        Run the complete trading system analysis.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            asset_classes: Asset classes to analyze
            exchanges: Exchanges to include
            custom_symbols: Custom symbols to analyze
            
        Returns:
            Complete analysis results
        """
        # Set default parameters
        asset_classes = asset_classes or ["equities", "etfs"]
        exchanges = exchanges or ["NYSE", "NASDAQ"]
        custom_symbols = custom_symbols or []
        
        self.logger.info(f"Starting trading system analysis from {start_date} to {end_date}")
        
        results = {}
        
        try:
            # Execute analysis pipeline in stages
            universe_result = await self._run_data_universe_stage(
                start_date, end_date, asset_classes, exchanges, custom_symbols
            )
            results['data_universe'] = universe_result
            
            technical_result = await self._run_technical_analysis_stage(universe_result)
            if technical_result.success:
                results['technical_analysis'] = technical_result
            
            ml_result = await self._run_ml_ensemble_stage(universe_result, technical_result)
            if ml_result and ml_result.success:
                results['ml_ensemble'] = ml_result
            
            strategy_results = await self._run_strategy_stages(universe_result, technical_result, results)
            results.update(strategy_results)
            
            synthesis_result = await self._run_signal_synthesis_stage(results)
            if synthesis_result and synthesis_result.success:
                results['signal_synthesis'] = synthesis_result
                
                risk_result = await self._run_risk_modeling_stage(synthesis_result)
                if risk_result and risk_result.success:
                    results['risk_modeling'] = risk_result
            
            final_recommendations = await self._run_final_recommendation_stage(results)
            if final_recommendations:
                results['final_recommendations'] = final_recommendations
                
            return self._create_analysis_summary(results)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {"error": str(e), "partial_results": results}
    
    async def _run_data_universe_stage(self, start_date, end_date, asset_classes, exchanges, custom_symbols):
        """Run data universe and preprocessing stage."""
        self.logger.info("Step 1: Data Universe & Preprocessing")
        
        universe_inputs = {
            "start_date": start_date,
            "end_date": end_date,
            "asset_classes": asset_classes,
            "exchanges": exchanges,
            "custom_symbols": custom_symbols
        }
        
        universe_result = await self.agents['data_universe'].safe_execute(universe_inputs)
        if not universe_result.success:
            raise Exception(f"Data universe failed: {universe_result.error_message}")
        
        symbols_count = universe_result.data['metadata']['symbols_count']
        self.logger.info(f"Universe processing completed with {symbols_count} symbols")
        return universe_result
    
    async def _run_technical_analysis_stage(self, universe_result):
        """Run technical analysis and feature engineering stage."""
        self.logger.info("Step 2: Feature Engineering & Technical Analysis")
        
        technical_inputs = {
            "feature_matrix": universe_result.data["feature_matrix"],
            "symbols": list(universe_result.data["universe"]),
            "add_patterns": True,
            "add_microstructure": False
        }
        
        technical_result = await self.agents['technical_analysis'].safe_execute(technical_inputs)
        if not technical_result.success:
            self.logger.warning(f"Technical analysis failed: {technical_result.error_message}")
        else:
            features_added = technical_result.data['feature_summary']['total_features_added']
            self.logger.info(f"Technical analysis completed with {features_added} features added")
        
        return technical_result
    
    async def _run_ml_ensemble_stage(self, universe_result, technical_result):
        """Run ML ensemble stage if technical analysis succeeded."""
        if not technical_result.success:
            return None
            
        self.logger.info("Step 3: ML Model Ensemble")
        
        ml_inputs = {
            "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
            "target_variable": "forward_return_1d",
            "train_models": True,
            "prediction_mode": "regression",
            "symbols": list(universe_result.data["universe"])
        }
        
        ml_result = await self.agents['ml_ensemble'].safe_execute(ml_inputs)
        if not ml_result.success:
            self.logger.warning(f"ML ensemble failed: {ml_result.error_message}")
        else:
            models_trained = ml_result.metadata['models_trained']
            self.logger.info(f"ML ensemble completed with {models_trained} models trained")
        
        return ml_result
    
    async def _run_strategy_stages(self, universe_result, technical_result, results):
        """Run all strategy analysis stages."""
        strategy_results = {}
        
        if technical_result.success:
            # Run strategies that depend on technical analysis
            momentum_result = await self._run_momentum_strategy(universe_result, technical_result, results)
            if momentum_result and momentum_result.success:
                strategy_results['momentum_strategy'] = momentum_result
            
            stat_arb_result = await self._run_stat_arb_strategy(universe_result, technical_result)
            if stat_arb_result and stat_arb_result.success:
                strategy_results['stat_arb'] = stat_arb_result
            
            cross_asset_result = await self._run_cross_asset_strategy(universe_result, technical_result)
            if cross_asset_result and cross_asset_result.success:
                strategy_results['cross_asset'] = cross_asset_result
        
        # Run strategies that don't require technical analysis
        event_driven_result = await self._run_event_driven_strategy(universe_result)
        if event_driven_result and event_driven_result.success:
            strategy_results['event_driven'] = event_driven_result
        
        options_result = await self._run_options_strategy(universe_result)
        if options_result and options_result.success:
            strategy_results['options_strategy'] = options_result
        
        return strategy_results
    
    async def _run_momentum_strategy(self, universe_result, technical_result, results):
        """Run momentum strategy analysis."""
        self.logger.info("Step 4A: Momentum Strategy Analysis")
        
        momentum_inputs = {
            "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
            "ml_predictions": results.get('ml_ensemble', {}).get('data', {}) if 'ml_ensemble' in results else {},
            "symbols": list(universe_result.data["universe"]),
            "current_positions": {}
        }
        
        momentum_result = await self.agents['momentum_strategy'].safe_execute(momentum_inputs)
        if not momentum_result.success:
            self.logger.warning(f"Momentum strategy failed: {momentum_result.error_message}")
        else:
            recs_count = momentum_result.metadata.get('recommendations_count', 0)
            self.logger.info(f"Momentum strategy completed with {recs_count} recommendations")
        
        return momentum_result
    
    async def _run_stat_arb_strategy(self, universe_result, technical_result):
        """Run statistical arbitrage strategy."""
        self.logger.info("Step 4B: Statistical Arbitrage Analysis")
        
        stat_arb_inputs = {
            "enhanced_feature_matrix": technical_result.data["enhanced_feature_matrix"],
            "symbols": list(universe_result.data["universe"])[:20],  # Limit for pairs analysis
            "current_positions": {},
            "lookback_window": 252
        }
        
        stat_arb_result = await self.agents['stat_arb'].safe_execute(stat_arb_inputs)
        if not stat_arb_result.success:
            self.logger.warning(f"Statistical arbitrage failed: {stat_arb_result.error_message}")
        else:
            pairs_count = stat_arb_result.metadata.get('pairs_identified', 0)
            self.logger.info(f"Statistical arbitrage completed with {pairs_count} pairs identified")
        
        return stat_arb_result
    
    async def _run_event_driven_strategy(self, universe_result):
        """Run event-driven strategy analysis."""
        self.logger.info("Step 4C: Event-Driven Analysis")
        
        event_driven_inputs = {
            "symbols": list(universe_result.data["universe"])[:10],  # Limit for news analysis
            "news_data": [],  # Would be populated with real news data
            "earnings_calendar": {},  # Would be populated with earnings data
            "corporate_events": []  # Would be populated with corporate events
        }
        
        event_driven_result = await self.agents['event_driven'].safe_execute(event_driven_inputs)
        if not event_driven_result.success:
            self.logger.warning(f"Event-driven strategy failed: {event_driven_result.error_message}")
        else:
            earnings_events = event_driven_result.metadata.get('earnings_events', 0)
            corporate_events = event_driven_result.metadata.get('corporate_events', 0)
            events_count = earnings_events + corporate_events
            self.logger.info(f"Event-driven analysis completed with {events_count} events analyzed")
        
        return event_driven_result
    
    async def _run_options_strategy(self, universe_result):
        """Run options strategy analysis."""
        self.logger.info("Step 4D: Options Strategy Analysis")
        
        options_inputs = {
            "symbols": list(universe_result.data["universe"])[:5],  # Limit for options analysis
            "options_chain": [],  # Would be populated with real options data
            "market_view": {"direction": "neutral", "volatility_expectation": "stable"},
            "risk_constraints": {"max_loss_per_trade": 1000}
        }
        
        options_result = await self.agents['options_strategy'].safe_execute(options_inputs)
        if not options_result.success:
            self.logger.warning(f"Options strategy failed: {options_result.error_message}")
        else:
            strategies_count = options_result.metadata.get('strategies_generated', 0)
            self.logger.info(f"Options strategy completed with {strategies_count} strategies generated")
        
        return options_result
    
    async def _run_cross_asset_strategy(self, universe_result, technical_result):
        """Run cross-asset strategy analysis."""
        self.logger.info("Step 4E: Cross-Asset Analysis")
        
        cross_asset_inputs = {
            "price_data": universe_result.data["feature_matrix"][list(universe_result.data["universe"])],
            "symbols": list(universe_result.data["universe"])[:15],  # Limit for cross-asset analysis
            "current_portfolio": {},
            "risk_constraints": {"risk_tolerance": "moderate"}
        }
        
        cross_asset_result = await self.agents['cross_asset'].safe_execute(cross_asset_inputs)
        if not cross_asset_result.success:
            self.logger.warning(f"Cross-asset analysis failed: {cross_asset_result.error_message}")
        else:
            assets_analyzed = cross_asset_result.metadata.get('assets_analyzed', 0)
            self.logger.info(f"Cross-asset analysis completed with {assets_analyzed} assets analyzed")
        
        return cross_asset_result
    
    async def _run_signal_synthesis_stage(self, results):
        """Run signal synthesis and consensus validation."""
        if 'momentum_strategy' not in results or not results['momentum_strategy'].success:
            return None
            
        self.logger.info("Step 5: Signal Synthesis & Consensus Validation")
        
        # Collect all strategy signals for synthesis
        strategy_signals = {}
        if 'momentum_strategy' in results:
            strategy_signals['momentum'] = results['momentum_strategy'].data
        
        if 'ml_ensemble' in results:
            strategy_signals['ml_ensemble'] = results['ml_ensemble'].data
        
        synthesis_inputs = self._build_synthesis_inputs(strategy_signals)
        
        synthesis_result = await self.agents['signal_synthesis'].safe_execute(synthesis_inputs)
        if not synthesis_result.success:
            self.logger.warning(f"Signal synthesis failed: {synthesis_result.error_message}")
        else:
            final_recs = synthesis_result.data.get('final_recommendations', [])
            self.logger.info(f"Signal synthesis completed with {len(final_recs)} consensus-validated recommendations")
        
        return synthesis_result
    
    async def _run_risk_modeling_stage(self, synthesis_result):
        """Run risk modeling and validation."""
        final_recs = synthesis_result.data.get('final_recommendations', [])
        if not final_recs:
            return None
            
        self.logger.info("Step 6: Advanced Risk Modeling & Multi-layer Validation")
        
        risk_inputs = self._build_risk_inputs(final_recs)
        
        risk_result = await self.agents['risk_modeling'].safe_execute(risk_inputs)
        if not risk_result.success:
            self.logger.warning(f"Risk modeling failed: {risk_result.error_message}")
        else:
            risk_validated_recs = risk_result.data.get('risk_validated_recommendations', [])
            consensus_achieved = risk_result.metadata.get('consensus_achieved', False)
            self.logger.info(f"Risk modeling completed: {len(risk_validated_recs)} risk-validated recommendations, consensus: {consensus_achieved}")
        
        return risk_result
    
    async def _run_final_recommendation_stage(self, results):
        """Generate final recommendations with cross-validation."""
        self.logger.info("Step 7: Final Recommendations with Cross-Validation")
        
        # Collect all agent outputs for final recommendation generation
        agent_outputs = self._collect_agent_outputs(results)
        
        if not agent_outputs:
            self.logger.warning("No agent outputs available for final recommendations")
            return None
        
        recommendation_inputs = self._build_recommendation_inputs(agent_outputs)
        
        final_result = await self.agents['recommendation'].safe_execute(recommendation_inputs)
        if not final_result.success:
            self.logger.warning(f"Final recommendations failed: {final_result.error_message}")
        else:
            final_recommendations = final_result.data.get('recommendations', [])
            self.logger.info(f"Final recommendations completed with {len(final_recommendations)} validated recommendations")
        
        return final_result
    
    def _build_synthesis_inputs(self, strategy_signals):
        """Build inputs for signal synthesis stage."""
        return {
            "strategy_signals": strategy_signals,
            "market_data": {
                "volatility_indicators": {
                    "current_volatility": 0.2,
                    "historical_volatility": 0.18
                },
                "trend_indicators": {
                    "trend_strength": 0.6,
                    "trend_direction": 1
                }
            },
            "risk_constraints": {
                "max_single_position": 0.1,
                "max_portfolio_risk": 0.02,
                "max_sector_exposure": 0.20,
                "max_total_leverage": 1.0
            },
            "portfolio_context": {
                "positions": {},
                "total_capital": 1000000
            }
        }
    
    def _build_risk_inputs(self, final_recs):
        """Build inputs for risk modeling stage."""
        return {
            "recommendations": final_recs,
            "portfolio_data": {
                "positions": {},
                "weights": {rec["symbol"]: rec.get("position_size", 0) for rec in final_recs},
                "total_capital": 1000000
            },
            "market_data": {
                "volatility_data": {"current_volatility": 0.2, "historical_volatility": 0.18},
                "correlation_data": {"average_correlation": 0.5}
            },
            "risk_constraints": {
                "max_var_95": -0.03,
                "max_volatility": 0.15,
                "max_single_position": 0.1,
                "max_sector_exposure": 0.25,
                "max_portfolio_risk": 0.02
            }
        }
    
    def _collect_agent_outputs(self, results):
        """Collect outputs from all agents for final recommendations."""
        agent_outputs = {}
        
        # Include all successful agent results
        for agent_name, result in results.items():
            if hasattr(result, 'success') and result.success:
                agent_outputs[agent_name] = result.data
            elif isinstance(result, dict):
                agent_outputs[agent_name] = result
        
        return agent_outputs
    
    def _build_recommendation_inputs(self, agent_outputs):
        """Build inputs for final recommendation stage."""
        return {
            "agent_outputs": agent_outputs,
            "aggregation_method": "weighted_consensus",
            "confidence_threshold": 0.6,
            "diversification_constraints": {
                "max_correlation": 0.7,
                "max_sector_weight": 0.3,
                "min_positions": 5
            },
            "risk_management": {
                "position_sizing_method": "equal_risk_contribution",
                "max_position_size": 0.1,
                "stop_loss_threshold": -0.05
            }
        }
    
    def _create_analysis_summary(self, results):
        """Create a comprehensive analysis summary."""
        summary = {
            "analysis_complete": True,
            "timestamp": datetime.now().isoformat(),
            "stages_completed": list(results.keys()),
            "summary_statistics": self._calculate_summary_statistics(results),
            "recommendations_summary": self._summarize_recommendations(results),
            "risk_assessment": self._summarize_risk_assessment(results),
            "full_results": results
        }
        
        self.logger.info(f"Analysis completed successfully with {len(summary['stages_completed'])} stages")
        return summary
    
    def _calculate_summary_statistics(self, results):
        """Calculate summary statistics across all stages."""
        stats = {
            "total_symbols_analyzed": 0,
            "total_strategies_executed": 0,
            "total_recommendations": 0,
            "average_confidence": 0.0
        }
        
        if 'data_universe' in results:
            stats["total_symbols_analyzed"] = results['data_universe'].data.get('metadata', {}).get('symbols_count', 0)
        
        # Count successful strategy executions
        strategy_keys = ['momentum_strategy', 'stat_arb', 'event_driven', 'options_strategy', 'cross_asset']
        stats["total_strategies_executed"] = sum(1 for key in strategy_keys if key in results and hasattr(results[key], 'success') and results[key].success)
        
        if 'final_recommendations' in results:
            final_recs = results['final_recommendations'].data.get('recommendations', [])
            stats["total_recommendations"] = len(final_recs)
            if final_recs:
                confidences = [rec.get('confidence', 0) for rec in final_recs if 'confidence' in rec]
                stats["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0
        
        return stats
    
    def _summarize_recommendations(self, results):
        """Summarize final recommendations."""
        if 'final_recommendations' not in results:
            return {"total": 0, "by_action": {}}
        
        recommendations = results['final_recommendations'].data.get('recommendations', [])
        
        summary = {
            "total": len(recommendations),
            "by_action": {},
            "top_recommendations": recommendations[:5] if recommendations else []
        }
        
        # Count by action type
        for rec in recommendations:
            action = rec.get('action', 'unknown')
            summary["by_action"][action] = summary["by_action"].get(action, 0) + 1
        
        return summary
    
    def _summarize_risk_assessment(self, results):
        """Summarize risk assessment results."""
        if 'risk_modeling' not in results:
            return {"available": False}
        
        risk_data = results['risk_modeling'].data
        
        return {
            "available": True,
            "portfolio_var": risk_data.get('portfolio_var', 0),
            "max_drawdown": risk_data.get('max_drawdown', 0),
            "sharpe_ratio": risk_data.get('sharpe_ratio', 0),
            "risk_score": risk_data.get('overall_risk_score', 0)
        }


# Usage example and main execution
async def main():
    """Main execution function for the trading system."""
    system = TradingSystem()
    
    # Define analysis parameters
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    
    # Run analysis
    results = await system.run_analysis(
        start_date=start_date,
        end_date=end_date,
        asset_classes=["equities"],
        exchanges=["NYSE", "NASDAQ"],
        custom_symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    )
    
    # Print summary
    if "error" not in results:
        print("="*80)
        print(" TRADING SYSTEM ANALYSIS COMPLETE")
        print("="*80)
        print(f"Stages completed: {len(results['stages_completed'])}")
        print(f"Total recommendations: {results['summary_statistics']['total_recommendations']}")
        print(f"Average confidence: {results['summary_statistics']['average_confidence']:.2f}")
    else:
        print(f"Analysis failed: {results['error']}")


if __name__ == "__main__":
    asyncio.run(main())