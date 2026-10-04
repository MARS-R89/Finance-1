/**
 * Indian Rupee (INR) Formatting & Fintech Utilities
 * Formats a numeric value into standard Indian currency representation.
 * Supports Lakhs (L) and Crores (Cr) abbreviations.
 * Examples:
 *   12500000 -> "₹ 1.25 Cr"
 *   450000   -> "₹ 4.50 L"
 *   75000    -> "₹ 75,000"
 */
export function formatINR(amount: number, compact = true): string {
  if (isNaN(amount) || amount === null || amount === undefined) {
    return '₹ 0';
  }
  const isNegative = amount < 0;
  const absAmount = Math.abs(amount);
  if (compact) {
    if (absAmount >= 10000000) {
      const cr = absAmount / 10000000;
      return `${isNegative ? '-' : ''}₹ ${cr.toFixed(2)} Cr`;
    }
    if (absAmount >= 100000) {
      const lakh = absAmount / 100000;
      return `${isNegative ? '-' : ''}₹ ${lakh.toFixed(2)} L`;
    }
  }
  // Standard Indian comma grouping: 1,23,456
  const parts = Math.round(absAmount).toString().split('');
  let lastThree = parts.slice(-3).join('');
  const otherNumbers = parts.slice(0, -3).join('');
  if (otherNumbers !== '') {
    lastThree = ',' + lastThree;
  }
  const formatted = otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ',') + lastThree;
  return `${isNegative ? '-' : ''}₹ ${formatted}`;
}

/**
 * Formats full numeric amount without L/Cr abbreviation for tooltips
 */
export function formatINRFull(amount: number): string {
  return formatINR(amount, false);
}

/**
 * Formats percentage e.g. 0.123 -> "12.3%"
 */
export function formatPercent(value: number, decimals = 1, isDecimal = true): string {
  const num = isDecimal ? value * 100 : value;
  return `${num.toFixed(decimals)}%`;
}

/**
 * Curated Fintech Color Palette for Assets & Categories
 */
export const ASSET_COLORS: Record<string, string> = {
  nifty_bank: '#3b82f6', // Blue
  nifty_it: '#06b6d4', // Cyan
  nifty_pharma: '#10b981', // Emerald
  nifty_fmcg: '#84cc16', // Lime
  nifty_auto: '#f59e0b', // Amber
  nifty_metal: '#ef4444', // Red
  nifty_energy: '#f97316', // Orange
  nifty_realty: '#8b5cf6', // Violet
  nifty_infra: '#6366f1', // Indigo
  gold: '#eab308', // Gold / Yellow
  mf_large_cap: '#14b8a6', // Teal
  mf_mid_cap: '#0284c7', // Sky
  mf_small_cap: '#ec4899', // Pink
  mf_flexi_cap: '#a855f7', // Purple
  mf_debt: '#64748b', // Slate
  nifty_50: '#38bdf8', // Sky Blue (Direct Stocks)
  bonds_fd: '#94a3b8', // Gray
};

export function getAssetColor(assetId: string, index = 0): string {
  if (ASSET_COLORS[assetId]) return ASSET_COLORS[assetId];
  const fallbackPalette = [
    '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
    '#06b6d4', '#ec4899', '#14b8a6', '#f97316'
  ];
  return fallbackPalette[index % fallbackPalette.length];
}
