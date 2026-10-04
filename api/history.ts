/**
 * Vercel Serverless Function / Express Handler: /api/history
 * Takes `?id=<asset_id | all>`
 * Returns monthly price/NAV series for up to 10 years, CAGR, Volatility,
 * Covariance & Correlation matrices, with Cache-Control and graceful fallback.
 */
import type { IncomingMessage, ServerResponse } from 'http';
import { getAssetHistory, getFullUniverseHistory } from '../src/engine/dataService.ts';

interface VercelRequest extends IncomingMessage {
  query?: Record<string, string | string[]>;
  url?: string;
}

interface VercelResponse extends ServerResponse {
  status(statusCode: number): this;
  json(jsonBody: any): this;
  setHeader(name: string, value: number | string | readonly string[]): this;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  res.setHeader('Cache-Control', 'public, max-age=86400, s-maxage=86400, stale-while-revalidate=43200');

  try {
    let id = 'all';
    if (req.query && req.query.id) {
      id = Array.isArray(req.query.id) ? req.query.id[0] : req.query.id;
    } else if (req.url && req.url.includes('?')) {
      const searchParams = new URL(req.url, 'http://localhost').searchParams;
      id = searchParams.get('id') || 'all';
    }

    if (id === 'all' || !id) {
      const fullUniverse = await getFullUniverseHistory();
      res.statusCode = 200;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(fullUniverse, null, 2));
      return;
    }

    const assetHistory = await getAssetHistory(id);
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');
    res.end(
      JSON.stringify(
        {
          asset: assetHistory,
          usingFallbackData: !!assetHistory.isFallback,
          timestamp: Date.now(),
        },
        null,
        2
      )
    );
  } catch (error: any) {
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json');
    res.end(
      JSON.stringify({
        error: 'Failed to process asset history',
        message: error?.message || 'Unknown error',
        usingFallbackData: true,
      })
    );
  }
}
