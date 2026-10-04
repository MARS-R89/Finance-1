/**
 * Asset Universe definitions for the Portfolio Simulator
 */
import { AssetDefinition } from './types.ts';

export const SECTOR_INDICES: AssetDefinition[] = [
  {
    id: 'nifty_bank',
    name: 'Nifty Bank',
    category: 'sector',
    tickerOrCode: '^NSEBANK',
    description: '12 most liquid and large Indian banking stocks (PSU and Private).',
    benchmark: 'Nifty Bank PR',
  },
  {
    id: 'nifty_it',
    name: 'Nifty IT',
    category: 'sector',
    tickerOrCode: '^CNXIT',
    description: 'Indian Information Technology leaders (TCS, Infosys, Wipro, HCL, TechM).',
    benchmark: 'Nifty IT TR',
  },
  {
    id: 'nifty_pharma',
    name: 'Nifty Pharma',
    category: 'sector',
    tickerOrCode: '^CNXPHARMA',
    description: 'Leading pharmaceutical and healthcare companies in India.',
    benchmark: 'Nifty Pharma TR',
  },
  {
    id: 'nifty_fmcg',
    name: 'Nifty FMCG',
    category: 'sector',
    tickerOrCode: '^CNXFMCG',
    description: 'Fast-Moving Consumer Goods sector (ITC, HUL, Nestle, Britannia).',
    benchmark: 'Nifty FMCG TR',
  },
  {
    id: 'nifty_auto',
    name: 'Nifty Auto',
    category: 'sector',
    tickerOrCode: '^CNXAUTO',
    description: 'Automobile manufacturers, auto ancillaries, and EV supply chain.',
    benchmark: 'Nifty Auto TR',
  },
  {
    id: 'nifty_metal',
    name: 'Nifty Metal',
    category: 'sector',
    tickerOrCode: '^CNXMETAL',
    description: 'Mining, basic metals, steel, and aluminum producers (Tata Steel, JSW).',
    benchmark: 'Nifty Metal TR',
  },
  {
    id: 'nifty_energy',
    name: 'Nifty Energy',
    category: 'sector',
    tickerOrCode: '^CNXENERGY',
    description: 'Petroleum, gas, power utilities, and renewable energy providers.',
    benchmark: 'Nifty Energy TR',
  },
  {
    id: 'nifty_realty',
    name: 'Nifty Realty',
    category: 'sector',
    tickerOrCode: '^CNXREALTY',
    description: 'Residential and commercial real estate developers and REITs.',
    benchmark: 'Nifty Realty TR',
  },
  {
    id: 'nifty_infra',
    name: 'Nifty Infra',
    category: 'sector',
    tickerOrCode: '^CNXINFRA',
    description: 'Infrastructure, capital goods, construction, ports, and logistics.',
    benchmark: 'Nifty Infrastructure TR',
  },
];

export const COMMODITY_ASSETS: AssetDefinition[] = [
  {
    id: 'gold',
    name: 'Gold',
    category: 'gold',
    tickerOrCode: 'GOLDBEES.NS', // Yahoo ticker for Nippon India ETF Gold BeES (INR)
    description: 'Physical gold commodity / Gold ETF hedge against inflation and volatility.',
    benchmark: 'Domestic Gold Price',
  },
];

export const MUTUAL_FUND_CATEGORIES: AssetDefinition[] = [
  {
    id: 'mf_large_cap',
    name: 'Large Cap Fund',
    category: 'mf',
    tickerOrCode: '120716', // Nippon India Large Cap Fund Direct-Growth
    description: 'Top 100 Indian companies by market capitalization with stable growth.',
    benchmark: 'NIFTY 100 TRI',
  },
  {
    id: 'mf_mid_cap',
    name: 'Mid Cap Fund',
    category: 'mf',
    tickerOrCode: '118989', // HDFC Mid-Cap Opportunities Fund Direct-Growth
    description: 'Companies ranked 101st to 250th with high growth potential.',
    benchmark: 'NIFTY Midcap 150 TRI',
  },
  {
    id: 'mf_small_cap',
    name: 'Small Cap Fund',
    category: 'mf',
    tickerOrCode: '118778', // Nippon India Small Cap Fund Direct-Growth
    description: 'Companies ranked 251st and above with hyper-growth and high volatility.',
    benchmark: 'NIFTY Smallcap 250 TRI',
  },
  {
    id: 'mf_flexi_cap',
    name: 'Flexi Cap Fund',
    category: 'mf',
    tickerOrCode: '122639', // Parag Parikh Flexi Cap Fund Direct-Growth
    description: 'Dynamic unconstrained allocation across Large, Mid, Small caps and international equities.',
    benchmark: 'NIFTY 500 TRI',
  },
  {
    id: 'mf_debt',
    name: 'Debt Fund',
    category: 'mf',
    tickerOrCode: '119062', // HDFC Short Term Debt Fund Direct-Growth
    description: 'High-quality sovereign and AAA corporate bonds with lower interest rate risk.',
    benchmark: 'CRISIL Short Duration Debt Index',
  },
];

export const DIRECT_STOCKS_ASSET: AssetDefinition = {
  id: 'nifty_50',
  name: 'Direct Stocks (Nifty 50 proxy)',
  category: 'stock',
  tickerOrCode: '^NSEI',
  description: 'Direct equity holdings represented via the flagship Nifty 50 index proxy.',
  benchmark: 'NIFTY 50',
};

export const BONDS_FD_DEFINITION: AssetDefinition = {
  id: 'bonds_fd',
  name: 'Bonds / Fixed Deposit',
  category: 'fixed_income',
  tickerOrCode: 'BONDS_FD',
  description: 'Fixed income with predictable coupon schedule, YTM, and credit default haircut modelling.',
};

export const ALL_VARIABLE_ASSETS: AssetDefinition[] = [
  ...SECTOR_INDICES,
  ...COMMODITY_ASSETS,
  ...MUTUAL_FUND_CATEGORIES,
  DIRECT_STOCKS_ASSET,
];

export const ALL_ASSET_DEFINITIONS: AssetDefinition[] = [
  ...ALL_VARIABLE_ASSETS,
  BONDS_FD_DEFINITION,
];

export const ASSET_MAP: Record<string, AssetDefinition> = ALL_ASSET_DEFINITIONS.reduce(
  (acc, asset) => {
    acc[asset.id] = asset;
    return acc;
  },
  {} as Record<string, AssetDefinition>
);
