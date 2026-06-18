import '../../models/ai_message.dart';

const bootMessages = [
  AiMessage(text: '>> INITIALIZING AI CORE', category: SentimentCategory.boot, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> LOADING MARKET MODELS', category: SentimentCategory.boot, zone: SentimentZone.leftHud),
  AiMessage(text: '>> CONNECTING TO EXCHANGES', category: SentimentCategory.boot, zone: SentimentZone.rightHud),
  AiMessage(text: '>> SYNCHRONIZING PRICE FEEDS', category: SentimentCategory.boot, zone: SentimentZone.leftArm),
  AiMessage(text: '>> STARTING NEURAL NETWORK', category: SentimentCategory.boot, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> VERIFYING DATA STREAM', category: SentimentCategory.boot, zone: SentimentZone.rightArm),
  AiMessage(text: '>> SYSTEM ONLINE', category: SentimentCategory.boot, zone: SentimentZone.chest, priority: 2, isCritical: true),
];

const marketMessages = [
  AiMessage(text: '>> SCANNING FOR OPPORTUNITIES', category: SentimentCategory.market, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> ANALYZING LIQUIDITY', category: SentimentCategory.market, zone: SentimentZone.leftHud),
  AiMessage(text: '>> CALCULATING RISK', category: SentimentCategory.market, zone: SentimentZone.rightHud),
  AiMessage(text: '>> SEARCHING FOR ENTRY', category: SentimentCategory.market, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> ANALYZING VOLATILITY', category: SentimentCategory.market, zone: SentimentZone.leftArm),
  AiMessage(text: '>> CHECKING TREND STRENGTH', category: SentimentCategory.market, zone: SentimentZone.rightArm),
  AiMessage(text: '>> MONITORING PRICE ACTION', category: SentimentCategory.market, zone: SentimentZone.leftHud),
  AiMessage(text: '>> DETECTING MARKET STRUCTURE', category: SentimentCategory.market, zone: SentimentZone.rightHud),
  AiMessage(text: '>> MEASURING MOMENTUM', category: SentimentCategory.market, zone: SentimentZone.leftArm),
  AiMessage(text: '>> SCANNING ORDER FLOW', category: SentimentCategory.market, zone: SentimentZone.rightArm),
  AiMessage(text: '>> TREND: BULLISH', category: SentimentCategory.market, zone: SentimentZone.leftHud),
  AiMessage(text: '>> VOLATILITY INDEX: 0.82', category: SentimentCategory.market, zone: SentimentZone.rightHud),
  AiMessage(text: '>> LIQUIDITY POOL DETECTED', category: SentimentCategory.market, zone: SentimentZone.chest),
  AiMessage(text: '>> PATTERN: ASCENDING TRIANGLE', category: SentimentCategory.market, zone: SentimentZone.leftHud),
  AiMessage(text: '>> SUPPORT LEVEL: HOLDING', category: SentimentCategory.market, zone: SentimentZone.rightArm),
  AiMessage(text: '>> RESISTANCE: TESTING', category: SentimentCategory.market, zone: SentimentZone.leftArm),
  AiMessage(text: '>> WHALE ACTIVITY: LOW', category: SentimentCategory.market, zone: SentimentZone.rightHud),
  AiMessage(text: '>> CORRELATION: 0.74', category: SentimentCategory.market, zone: SentimentZone.lowerHud),
  AiMessage(text: '>> MARKET STATE: ACTIVE', category: SentimentCategory.market, zone: SentimentZone.chest),
];

const intelligenceMessages = [
  AiMessage(text: '>> UPDATING NEURAL MODEL', category: SentimentCategory.intelligence, zone: SentimentZone.chest),
  AiMessage(text: '>> OPTIMIZING STRATEGY', category: SentimentCategory.intelligence, zone: SentimentZone.leftHud),
  AiMessage(text: '>> LEARNING MARKET BEHAVIOR', category: SentimentCategory.intelligence, zone: SentimentZone.rightHud),
  AiMessage(text: '>> ADAPTIVE MODEL ACTIVE', category: SentimentCategory.intelligence, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> CONFIDENCE INCREASING', category: SentimentCategory.intelligence, zone: SentimentZone.leftArm),
  AiMessage(text: '>> PREDICTION ENGINE ONLINE', category: SentimentCategory.intelligence, zone: SentimentZone.rightArm),
  AiMessage(text: '>> MULTI-TIMEFRAME ANALYSIS', category: SentimentCategory.intelligence, zone: SentimentZone.leftHud),
  AiMessage(text: '>> SENTIMENT ENGINE ACTIVE', category: SentimentCategory.intelligence, zone: SentimentZone.rightHud),
  AiMessage(text: '>> SELF-CALIBRATING', category: SentimentCategory.intelligence, zone: SentimentZone.chest),
  AiMessage(text: '>> PATTERN MATCH FOUND', category: SentimentCategory.intelligence, zone: SentimentZone.lowerHud),
];

const tradingMessages = [
  AiMessage(text: '>> ENTRY CONFIRMED', category: SentimentCategory.trading, zone: SentimentZone.chest, priority: 2, isCritical: true),
  AiMessage(text: '>> EXECUTING BUY TRADE', category: SentimentCategory.trading, zone: SentimentZone.chest, priority: 2, isCritical: true),
  AiMessage(text: '>> EXECUTING SELL TRADE', category: SentimentCategory.trading, zone: SentimentZone.chest, priority: 2, isCritical: true),
  AiMessage(text: '>> STOP LOSS DEPLOYED', category: SentimentCategory.trading, zone: SentimentZone.rightHud),
  AiMessage(text: '>> TAKE PROFIT LOCKED', category: SentimentCategory.trading, zone: SentimentZone.leftHud),
  AiMessage(text: '>> POSITION OPENED', category: SentimentCategory.trading, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> POSITION CLOSED', category: SentimentCategory.trading, zone: SentimentZone.chest, priority: 1),
  AiMessage(text: '>> BREAK EVEN ENABLED', category: SentimentCategory.trading, zone: SentimentZone.leftArm),
  AiMessage(text: '>> SCALING POSITION', category: SentimentCategory.trading, zone: SentimentZone.rightArm),
  AiMessage(text: '>> ORDER EXECUTED', category: SentimentCategory.trading, zone: SentimentZone.lowerHud, priority: 1),
];

const protectionMessages = [
  AiMessage(text: '>> HEDGING MODE ACTIVATED', category: SentimentCategory.protection, zone: SentimentZone.chest, priority: 2, isCritical: true),
  AiMessage(text: '>> DRAWDOWN CONTROL ACTIVE', category: SentimentCategory.protection, zone: SentimentZone.leftHud),
  AiMessage(text: '>> CAPITAL SHIELD ENABLED', category: SentimentCategory.protection, zone: SentimentZone.rightHud),
  AiMessage(text: '>> VOLATILITY PROTECTION', category: SentimentCategory.protection, zone: SentimentZone.leftArm),
  AiMessage(text: '>> RISK LIMIT REACHED', category: SentimentCategory.protection, zone: SentimentZone.rightArm),
  AiMessage(text: '>> REDUCING EXPOSURE', category: SentimentCategory.protection, zone: SentimentZone.chest),
  AiMessage(text: '>> SAFE MODE ENABLED', category: SentimentCategory.protection, zone: SentimentZone.lowerHud),
  AiMessage(text: '>> RECOVERY ENGINE ACTIVE', category: SentimentCategory.protection, zone: SentimentZone.chest),
];

const aggressiveMessages = [
  AiMessage(text: '>> AGGRESSIVE SCALPING ENABLED', category: SentimentCategory.aggressive, zone: SentimentZone.chest, priority: 2, isCritical: true),
  AiMessage(text: '>> HIGH FREQUENCY MODE', category: SentimentCategory.aggressive, zone: SentimentZone.rightHud),
  AiMessage(text: '>> TARGET LOCKED', category: SentimentCategory.aggressive, zone: SentimentZone.chest, priority: 1, isCritical: true),
  AiMessage(text: '>> BREAKOUT DETECTED', category: SentimentCategory.aggressive, zone: SentimentZone.leftHud),
  AiMessage(text: '>> FAST EXECUTION', category: SentimentCategory.aggressive, zone: SentimentZone.rightArm),
  AiMessage(text: '>> PRECISION ENTRY', category: SentimentCategory.aggressive, zone: SentimentZone.leftArm),
  AiMessage(text: '>> MAXIMUM ACCURACY', category: SentimentCategory.aggressive, zone: SentimentZone.lowerHud),
  AiMessage(text: '>> SNIPER MODE ACTIVE', category: SentimentCategory.aggressive, zone: SentimentZone.chest, priority: 2),
];
