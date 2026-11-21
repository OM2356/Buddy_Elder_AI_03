import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import type { RiskLevel, SupportedLanguage } from '../shared/types.ts';
import { isVercelRuntime } from './db.ts';
import { getGeminiApiKey } from './env.ts';

export const DEFAULT_GEMINI_MODEL = 'gemini-4-hydrogen';

const VALID_GEMINI_MODEL_REGEX = /^(?:gemini|lyria|genie)-[a-z0-9][a-z0-9.-]{1,60}$/i;
let loggedInvalidModelWarning = false;

export function normalizeGeminiModelName(raw: string | undefined | null): {
  model: string;
  invalidOverride: boolean;
} {
  const trimmed = raw?.trim() ?? '';
  if (!trimmed) {
    return { model: DEFAULT_GEMINI_MODEL, invalidOverride: false };
  }

  const stripped = trimmed.replace(/^models\//i, '').trim();
  const hasDoubleDot = stripped.includes('..') || stripped.endsWith('.') || stripped.endsWith('-');
  if (!hasDoubleDot && VALID_GEMINI_MODEL_REGEX.test(stripped)) {
    return { model: stripped.toLowerCase(), invalidOverride: false };
  }

  if (!loggedInvalidModelWarning) {
    loggedInvalidModelWarning = true;
    console.warn(
      `[Buddy AI Config] Ignoring invalid GEMINI_MODEL environment override format; falling back to ${DEFAULT_GEMINI_MODEL}.`
    );
  }
  return { model: DEFAULT_GEMINI_MODEL, invalidOverride: true };
}

export function getConfiguredGeminiModel(): string {
  return normalizeGeminiModelName(process.env.GEMINI_MODEL).model;
}

export type AiServiceErrorCode =
  | 'MISSING_API_KEY'
  | 'AI_AUTH_ERROR'
  | 'AI_RATE_LIMIT'
  | 'AI_QUOTA_EXHAUSTED'
  | 'AI_TIMEOUT'
  | 'AI_MODEL_ERROR'
  | 'AI_SAFETY_BLOCK'
  | 'AI_EMPTY_RESPONSE'
  | 'AI_NETWORK_ERROR'
  | 'UPSTREAM_ERROR';

export interface AiServiceErrorMeta {
  retryAfterSeconds?: number;
  geminiErrorCode?: string;
  rateLimitSource?: 'gemini_provider';
  quotaExhausted?: boolean;
  retryCount?: number;
  model?: string;
}

export class AiServiceError extends Error {
  statusCode: number;
  code: AiServiceErrorCode;
  upstreamType?: string;
  retryAfterSeconds?: number;
  geminiErrorCode?: string;
  rateLimitSource?: 'gemini_provider';
  quotaExhausted?: boolean;
  retryCount?: number;
  model?: string;

  constructor(
    message: string,
    statusCode: number,
    code: AiServiceErrorCode,
    upstreamType?: string,
    meta: AiServiceErrorMeta = {}
  ) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.upstreamType = upstreamType;
    this.retryAfterSeconds = meta.retryAfterSeconds;
    this.geminiErrorCode = meta.geminiErrorCode;
    this.rateLimitSource = meta.rateLimitSource;
    this.quotaExhausted = meta.quotaExhausted;
    this.retryCount = meta.retryCount;
    this.model = meta.model;
  }
}

function getClient(): GoogleGenAI {
  const apiKey = getGeminiApiKey();
  if (!apiKey) {
    throw new AiServiceError(
      'AI Assistant is not configured yet (GEMINI_API_KEY is missing in environment secrets). Please add GEMINI_API_KEY in Settings > Secrets and try again.',
      503,
      'MISSING_API_KEY',
      'missing_api_key'
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

export function getAiStatus(): { configured: boolean; model: string; modelWarning?: string } {
  const normalized = normalizeGeminiModelName(process.env.GEMINI_MODEL);
  return {
    configured: Boolean(getGeminiApiKey()),
    model: normalized.model,
    ...(normalized.invalidOverride
      ? {
          modelWarning:
            'GEMINI_MODEL contains an unrecognized value and was ignored; using default gemini-4-hydrogen. Check GEMINI_MODEL and GEMINI_API_KEY in environment secrets.',
        }
      : {}),
  };
}

function languageInstruction(language: SupportedLanguage): string {
  switch (language) {
    case 'hi':
      return 'Respond in warm, simple, clear Hindi (Devanagari script) suitable for an older adult. Keep technical terms simple.';
    case 'mr':
      return 'Respond in warm, simple, clear Marathi (Devanagari script) suitable for an older adult. Keep technical terms simple.';
    case 'hinglish':
      return 'Respond in warm, friendly Hinglish (conversational Hindi written in Latin/English letters mixed with simple everyday English words) suitable for an older adult.';
    default:
      return 'Respond in warm, patient, easy-to-read English suitable for an older adult. Use short paragraphs and clear numbered steps.';
  }
}

function parseRetrySeconds(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return Math.ceil(value);
  }
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  const secMatch = trimmed.match(/^(\d+(?:\.\d+)?)s$/i);
  if (secMatch) {
    const num = Number(secMatch[1]);
    return num > 0 ? Math.ceil(num) : undefined;
  }
  const intNum = Number(trimmed);
  if (Number.isFinite(intNum) && intNum > 0) {
    return Math.ceil(intNum);
  }
  return undefined;
}

function extractErrorContext(err: unknown): {
  friendly: string;
  combinedLower: string;
  numericCode?: number;
  geminiErrorCode?: string;
  retryAfterSeconds?: number;
  isDailyQuotaExhausted: boolean;
} {
  const raw = err instanceof Error ? err.message : String(err ?? 'Unknown AI provider error');
  const errObj = err as {
    status?: unknown;
    code?: unknown;
    headers?: { get?: (name: string) => string | null } | Record<string, string | undefined>;
    response?: { headers?: { get?: (name: string) => string | null } | Record<string, string | undefined> };
  };
  const sdkStatus = typeof errObj?.status === 'number' ? errObj.status : undefined;
  const sdkCode = errObj?.code;

  let jsonMessage: string | undefined;
  let jsonCode: number | undefined;
  let jsonStatus: string | undefined;
  let detailRetrySeconds: number | undefined;
  let detailQuotaText = '';

  const jsonStart = raw.indexOf('{');
  if (jsonStart !== -1) {
    try {
      const parsed = JSON.parse(raw.slice(jsonStart)) as {
        error?: {
          message?: string;
          status?: string;
          code?: number;
          details?: Array<{
            retryDelay?: string;
            '@type'?: string;
            violations?: Array<{ quotaId?: string; quotaMetric?: string; description?: string }>;
          }>;
        };
      };
      if (typeof parsed?.error?.message === 'string' && parsed.error.message.trim()) {
        jsonMessage = parsed.error.message.trim();
      }
      if (typeof parsed?.error?.code === 'number') {
        jsonCode = parsed.error.code;
      }
      if (typeof parsed?.error?.status === 'string') {
        jsonStatus = parsed.error.status;
      }
      if (Array.isArray(parsed?.error?.details)) {
        for (const d of parsed.error.details) {
          if (d?.retryDelay) {
            const parsedSec = parseRetrySeconds(d.retryDelay);
            if (parsedSec !== undefined) detailRetrySeconds = parsedSec;
          }
          if (Array.isArray(d?.violations)) {
            detailQuotaText +=
              ' ' +
              d.violations
                .map((v) => `${v?.quotaId ?? ''} ${v?.quotaMetric ?? ''} ${v?.description ?? ''}`)
                .join(' ');
          }
        }
      }
    } catch {
      // not JSON
    }
  }

  // Also check HTTP Retry-After header if exposed on SDK error
  let headerRetrySeconds: number | undefined;
  const hdrs = errObj?.headers ?? errObj?.response?.headers;
  if (hdrs) {
    const rawHdr =
      typeof (hdrs as { get?: (n: string) => string | null }).get === 'function'
        ? (hdrs as { get: (n: string) => string | null }).get('retry-after')
        : (hdrs as Record<string, string | undefined>)['retry-after'] ??
          (hdrs as Record<string, string | undefined>)['Retry-After'];
    headerRetrySeconds = parseRetrySeconds(rawHdr);
  }

  // Also parse "Please retry in 34.5s" from Gemini message text
  let textRetrySeconds: number | undefined;
  const retryTextMatch = raw.match(/retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b/i);
  if (retryTextMatch) {
    const num = Number(retryTextMatch[1]);
    if (Number.isFinite(num) && num > 0) textRetrySeconds = Math.ceil(num);
  }

  const retryAfterSeconds = headerRetrySeconds ?? detailRetrySeconds ?? textRetrySeconds;
  const friendly = jsonMessage || raw;
  const numericCode = jsonCode ?? sdkStatus ?? (typeof sdkCode === 'number' ? sdkCode : undefined);
  const combinedLower = [
    raw,
    friendly,
    detailQuotaText,
    sdkStatus !== undefined ? String(sdkStatus) : '',
    sdkCode !== undefined ? String(sdkCode) : '',
    jsonCode !== undefined ? String(jsonCode) : '',
    jsonStatus ?? '',
  ]
    .join(' ')
    .toLowerCase();

  const isDailyQuotaExhausted =
    combinedLower.includes('quota_exceeded') ||
    combinedLower.includes('perday') ||
    combinedLower.includes('per_day') ||
    combinedLower.includes('requestsperday') ||
    combinedLower.includes('daily quota') ||
    combinedLower.includes('daily limit') ||
    combinedLower.includes('limit: 0') ||
    (combinedLower.includes('billing') && combinedLower.includes('quota')) ||
    (retryAfterSeconds !== undefined && retryAfterSeconds > 60);

  const geminiErrorCode =
    jsonStatus ||
    (typeof sdkCode === 'string' && sdkCode) ||
    (numericCode !== undefined ? String(numericCode) : undefined);

  return {
    friendly,
    combinedLower,
    numericCode,
    geminiErrorCode,
    retryAfterSeconds,
    isDailyQuotaExhausted,
  };
}

export function classifyGeminiError(
  err: unknown,
  modelName = getConfiguredGeminiModel(),
  retryCount = 0
): AiServiceError {
  if (err instanceof AiServiceError) {
    if (err.retryCount === undefined) err.retryCount = retryCount;
    if (!err.model) err.model = modelName;
    return err;
  }
  const {
    friendly,
    combinedLower,
    numericCode,
    geminiErrorCode,
    retryAfterSeconds,
    isDailyQuotaExhausted,
  } = extractErrorContext(err);

  if (
    numericCode === 504 ||
    combinedLower.includes('etimedout') ||
    combinedLower.includes('timed out') ||
    combinedLower.includes('timeout')
  ) {
    return new AiServiceError(
      'Buddy Elder AI took too long to respond. Please tap Retry in a moment.',
      504,
      'AI_TIMEOUT',
      'timeout',
      { geminiErrorCode, retryCount, model: modelName }
    );
  }
  if (
    numericCode === 429 ||
    combinedLower.includes('429') ||
    combinedLower.includes('resource_exhausted') ||
    combinedLower.includes('quota') ||
    combinedLower.includes('rate limit')
  ) {
    if (isDailyQuotaExhausted) {
      return new AiServiceError(
        `Buddy AI is temporarily unavailable because today's AI service quota for ${modelName} has been reached. You can still use Is This Safe?, Learn Tech, Daily Check-In, and Reminders right now.`,
        429,
        'AI_QUOTA_EXHAUSTED',
        'quota_exhausted',
        {
          retryAfterSeconds,
          geminiErrorCode: geminiErrorCode || 'RESOURCE_EXHAUSTED',
          rateLimitSource: 'gemini_provider',
          quotaExhausted: true,
          retryCount,
          model: modelName,
        }
      );
    }
    const waitHint = retryAfterSeconds ?? 15;
    return new AiServiceError(
      `Buddy is helping many people right now. Please wait ${waitHint} seconds before trying again.`,
      429,
      'AI_RATE_LIMIT',
      'rate_limit',
      {
        retryAfterSeconds: waitHint,
        geminiErrorCode: geminiErrorCode || 'RESOURCE_EXHAUSTED',
        rateLimitSource: 'gemini_provider',
        quotaExhausted: false,
        retryCount,
        model: modelName,
      }
    );
  }
  if (
    numericCode === 401 ||
    numericCode === 403 ||
    combinedLower.includes('api_key_invalid') ||
    combinedLower.includes('api key not valid') ||
    combinedLower.includes('invalid api key') ||
    combinedLower.includes('permission_denied') ||
    combinedLower.includes('unauthenticated') ||
    combinedLower.includes('"code":401') ||
    combinedLower.includes('"code":403')
  ) {
    return new AiServiceError(
      'The configured GEMINI_API_KEY was rejected by the AI provider. Please verify GEMINI_API_KEY in server environment secrets.',
      502,
      'AI_AUTH_ERROR',
      'auth_error'
    );
  }
  if (
    numericCode === 404 ||
    combinedLower.includes('not_found') ||
    combinedLower.includes('"code":404') ||
    combinedLower.includes('unexpected model name format') ||
    (combinedLower.includes('invalid_argument') && combinedLower.includes('model')) ||
    (combinedLower.includes('model') &&
      (combinedLower.includes('not found') ||
        combinedLower.includes('not supported') ||
        combinedLower.includes('invalid') ||
        combinedLower.includes('unexpected')))
  ) {
    return new AiServiceError(
      `Configured Gemini model "${modelName}" is unavailable or unsupported for this API key. Set GEMINI_MODEL to a supported model or verify API access.`,
      502,
      'AI_MODEL_ERROR',
      'invalid_or_unavailable_model'
    );
  }
  if (
    combinedLower.includes('fetch failed') ||
    combinedLower.includes('enotfound') ||
    combinedLower.includes('econnrefused') ||
    combinedLower.includes('econnreset') ||
    combinedLower.includes('ehostunreach') ||
    combinedLower.includes('socket hang up') ||
    combinedLower.includes('network')
  ) {
    return new AiServiceError(
      'Unable to reach the Gemini AI service due to a network connection issue. Please try again shortly.',
      502,
      'AI_NETWORK_ERROR',
      'network_error'
    );
  }
  if (
    numericCode === 503 ||
    combinedLower.includes('unavailable') ||
    combinedLower.includes('overloaded') ||
    combinedLower.includes('high demand')
  ) {
    return new AiServiceError(
      'Buddy Elder AI is experiencing temporary high demand. Please tap Retry in a few seconds.',
      503,
      'UPSTREAM_ERROR',
      'service_unavailable'
    );
  }

  return new AiServiceError(
    `Unable to reach Buddy AI right now: ${friendly}`,
    502,
    'UPSTREAM_ERROR',
    'upstream_error'
  );
}

export function getRetryDelayMs(err: unknown, attempt: number): number | null {
  const ctx = extractErrorContext(err);
  // Never retry timeouts, aborts, or daily/hard quota exhaustion
  if (
    ctx.combinedLower.includes('etimedout') ||
    ctx.combinedLower.includes('aborterror') ||
    ctx.combinedLower.includes('aborted') ||
    ctx.isDailyQuotaExhausted
  ) {
    return null;
  }

  // Transient 429: only retry automatically if Retry-After is absent or <= 5s
  if (
    ctx.numericCode === 429 ||
    ctx.combinedLower.includes('resource_exhausted') ||
    ctx.combinedLower.includes('rate limit')
  ) {
    if (ctx.retryAfterSeconds !== undefined) {
      if (ctx.retryAfterSeconds > 5) return null;
      return ctx.retryAfterSeconds * 1000 + Math.floor(Math.random() * 250);
    }
    const base = 700 * Math.pow(2, attempt - 1);
    const jitter = Math.floor(Math.random() * 350);
    return base + jitter;
  }

  const isTransientNetOr503 =
    ctx.numericCode === 503 ||
    ctx.combinedLower.includes('unavailable') ||
    ctx.combinedLower.includes('econnreset') ||
    ctx.combinedLower.includes('fetch failed');

  if (!isTransientNetOr503) return null;
  const base = 600 * Math.pow(2, attempt - 1);
  const jitter = Math.floor(Math.random() * 300);
  return base + jitter;
}

function withAttemptTimeout<T>(fn: (signal: AbortSignal) => Promise<T>, timeoutMs = 25000): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  return Promise.race([
    fn(controller.signal),
    new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error('ETIMEDOUT: AI service took too long to respond'));
      }, timeoutMs);
    }),
  ]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

async function withAiRetry<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  maxAttempts = 2
): Promise<{ result: T; retryCount: number }> {
  let lastErr: unknown;
  let retryCount = 0;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await withAttemptTimeout(fn, 25000);
      return { result, retryCount };
    } catch (err) {
      lastErr = err;
      const delayMs = attempt < maxAttempts ? getRetryDelayMs(err, attempt) : null;
      if (delayMs === null) {
        const errorWithRetries =
          err && typeof err === 'object' ? Object.assign(err, { __buddyRetryCount: retryCount }) : err;
        throw errorWithRetries;
      }
      retryCount++;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastErr;
}

export function parseBuddyModelPayload(rawText: string): { reply: string; suggestions: string[] } {
  const trimmed = rawText.trim();
  if (!trimmed) {
    throw new AiServiceError(
      'Buddy Elder AI returned an empty response. Please try asking your question again.',
      502,
      'AI_EMPTY_RESPONSE',
      'empty_response'
    );
  }

  // Strip optional markdown JSON code fences
  const fenceMatch = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  const candidate = (fenceMatch ? fenceMatch[1] : trimmed).trim();

  if (candidate.startsWith('{')) {
    try {
      const parsed = JSON.parse(candidate) as { reply?: unknown; suggestions?: unknown };
      if (typeof parsed.reply === 'string' && parsed.reply.trim().length > 0) {
        const cleanSuggestions = Array.isArray(parsed.suggestions)
          ? parsed.suggestions.filter((s): s is string => typeof s === 'string' && s.trim().length > 0).slice(0, 3)
          : [];
        return { reply: parsed.reply.trim(), suggestions: cleanSuggestions };
      }
    } catch {
      // Fall through to check plain-text reply cleanly if non-JSON prose was returned
    }
  }

  // If the model returned valid human-readable text without JSON wrapper, return it cleanly
  if (!candidate.startsWith('{') && !candidate.startsWith('[')) {
    return { reply: candidate, suggestions: [] };
  }

  throw new AiServiceError(
    'Buddy Elder AI returned an unreadable response. Please tap Retry.',
    502,
    'AI_EMPTY_RESPONSE',
    'malformed_model_json'
  );
}

function extractCandidateText(response: {
  text?: string;
  promptFeedback?: { blockReason?: string };
  candidates?: Array<{
    finishReason?: string;
    content?: { parts?: Array<{ text?: string }> };
  }>;
}): string {
  const blockReason = response.promptFeedback?.blockReason;
  const finishReason = response.candidates?.[0]?.finishReason;
  const safetyBlocked =
    Boolean(blockReason) ||
    finishReason === 'SAFETY' ||
    finishReason === 'PROHIBITED_CONTENT' ||
    finishReason === 'BLOCKLIST' ||
    finishReason === 'RECITATION';

  if (safetyBlocked) {
    throw new AiServiceError(
      'I cannot answer that specific request safely. Please ask another question about daily technology, phone help, or staying safe online.',
      422,
      'AI_SAFETY_BLOCK',
      `safety_block:${blockReason || finishReason || 'unknown'}`
    );
  }

  const directText = typeof response.text === 'string' ? response.text.trim() : '';
  if (directText) return directText;

  const partText = (response.candidates?.[0]?.content?.parts ?? [])
    .map((p) => (typeof p.text === 'string' ? p.text : ''))
    .join('')
    .trim();
  return partText;
}

export async function generateBuddyReply(params: {
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  userMessage: string;
  language: SupportedLanguage;
  userName: string;
}): Promise<{ reply: string; suggestions: string[]; model: string; durationMs: number; retryCount: number }> {
  const startedAt = Date.now();
  const modelName = getConfiguredGeminiModel();
  const ai = getClient();

  const systemInstruction = [
    `You are Buddy Elder AI, a patient, respectful, and trustworthy digital companion helping ${params.userName}, a senior citizen.`,
    languageInstruction(params.language),
    'Give direct, practical, step-by-step instructions tailored specifically to the exact question asked (smartphones, WhatsApp, volume controls, bank/scam safety, Wi-Fi, internet, maps, and everyday tools).',
    'Keep responses concise (around 80-180 words unless detailed steps are needed) and easy to listen to when read aloud.',
    'STRICT SAFETY RULES:',
    '1. Never ask for or accept OTPs, passwords, bank PINs, card numbers, or Aadhaar/SSN details.',
    '2. Never diagnose medical conditions or prescribe treatments. If someone describes urgent medical pain, chest pain, breathing trouble, or immediate danger, gently advise calling local emergency services (112 / 108 / 911) or a doctor right away.',
    '3. Never claim to be emergency services, police, or a bank.',
    '4. If unsure, admit uncertainty honestly and suggest verifying with a trusted family member or official helpline.',
    '5. At the end, suggest 2 short helpful follow-up questions the senior can ask next.',
  ].join('\n');

  const validHistory = params.history
    .filter((m) => m && typeof m.content === 'string' && m.content.trim().length > 0)
    .slice(-12);

  const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];
  for (const item of validHistory) {
    const role = item.role === 'assistant' ? 'model' : 'user';
    const text = item.content.trim();
    if (contents.length === 0 && role === 'model') {
      continue;
    }
    const prev = contents[contents.length - 1];
    if (prev && prev.role === role) {
      prev.parts[0].text = `${prev.parts[0].text}\n\n${text}`;
    } else {
      contents.push({ role, parts: [{ text }] });
    }
  }

  const lastTurn = contents[contents.length - 1];
  if (lastTurn && lastTurn.role === 'user') {
    lastTurn.parts[0].text = `${lastTurn.parts[0].text}\n\n${params.userMessage}`;
  } else {
    contents.push({
      role: 'user',
      parts: [{ text: params.userMessage }],
    });
  }

  try {
    const { result: response, retryCount } = await withAiRetry((signal) =>
      ai.models.generateContent({
        model: modelName,
        contents,
        config: {
          abortSignal: signal,
          systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              reply: {
                type: Type.STRING,
                description: 'The warm, clear, step-by-step answer specifically answering the user question.',
              },
              suggestions: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: '2 short, helpful follow-up questions in the same language.',
              },
            },
            required: ['reply', 'suggestions'],
          },
        },
      })
    );

    const raw = extractCandidateText(
      response as unknown as {
        text?: string;
        promptFeedback?: { blockReason?: string };
        candidates?: Array<{ finishReason?: string; content?: { parts?: Array<{ text?: string }> } }>;
      }
    );
    const parsed = parseBuddyModelPayload(raw);
    return {
      reply: parsed.reply,
      suggestions: parsed.suggestions,
      model: modelName,
      durationMs: Date.now() - startedAt,
      retryCount,
    };
  } catch (err) {
    const retryCount =
      err && typeof err === 'object' && typeof (err as { __buddyRetryCount?: unknown }).__buddyRetryCount === 'number'
        ? (err as { __buddyRetryCount: number }).__buddyRetryCount
        : 0;
    throw classifyGeminiError(err, modelName, retryCount);
  }
}

export interface ScamAnalysisResult {
  riskLevel: RiskLevel;
  explanation: string;
  indicators: string[];
  recommendedActions: string[];
  requiresPause: boolean;
  source: 'ai' | 'heuristic_fallback' | 'heuristic_offline';
  fallbackReason?: AiServiceErrorCode;
}

export function runDeterministicScamChecks(
  content: string,
  inputType = 'message'
): {
  riskLevel: RiskLevel;
  indicators: string[];
  requiresPause: boolean;
} {
  const lower = content.toLowerCase();
  const normalizedType = inputType.trim().toLowerCase();
  const indicators: string[] = [];
  let score = 0;

  const criticalPatterns: Array<{ regex: RegExp; label: string; weight: number }> = [
    {
      regex: /\b(otp|one[\s-]?time[\s-]?password|verification code|cvv|atm pin|upi pin|mpin|netbanking password)\b/i,
      label: 'Requests OTP, PIN, password, or secret verification code',
      weight: 4,
    },
    {
      regex: /\b(anydesk|teamviewer|quicksupport|rustdesk|screen[\s-]?share|remote access)\b/i,
      label: 'Mentions remote screen-sharing software often abused in scams',
      weight: 4,
    },
    {
      regex:
        /\b(kyc.{0,30}(expire|suspend|block|update|verify|pending)|(expire|suspend|block|update|verify|complete).{0,25}\bkyc\b|(bank\s*)?account\b.{0,30}(block|freeze|suspend|close|deactivat|hold)|debit card.{0,20}(block|suspend)|credit card.{0,20}(block|suspend)|pan.{0,20}update|sim.{0,20}(block|suspend))\b/i,
      label: 'Threatens bank account, card, SIM, or KYC suspension/blocking',
      weight: 3,
    },
    {
      regex: /\b(click (this|the|below|here)?\s*link|tap (this|the|below)?\s*link|open (this|the)\s*link|verify (now|here|immediately)|login here)\b/i,
      label: 'Pressures you to click a verification link',
      weight: 2,
    },
    {
      regex: /\b(urgent|immediately|today only|right now|within \d+\s*(min|mins|minute|minutes|hour|hours|hr|hrs)|last warning|final notice|act now)\b/i,
      label: 'Creates urgent time pressure to rush your decision',
      weight: 2,
    },
    {
      regex: /\b(lottery|prize|jackpot|won\s*(rs|₹|\$|inr|usd|lakh|crore)|lucky draw|gift voucher|free gift)\b/i,
      label: 'Claims unexpected lottery, prize, or reward winnings',
      weight: 3,
    },
    {
      regex: /\b(digital arrest|cbi|ncb|customs parcel|arrest warrant|police case|cyber crime notice|tax notice)\b/i,
      label: 'Impersonates police, customs, tax, or government officials with threats',
      weight: 4,
    },
    {
      regex:
        /\b(pay\s*now|urgent\s*(transfer|payment)|send\s*money|deposit\s*fee|processing\s*fee|upi collect|scan qr to receive|received by mistake|refund claim|lost my phone|new number.*money)\b/i,
      label: 'Requests money transfer, fake refund/QR approval, or upfront fee',
      weight: 3,
    },
    {
      regex: /(bit\.ly|tinyurl\.com|cutt\.ly|rb\.gy|shorturl|is\.gd|t\.co|http:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/i,
      label: 'Contains shortened or raw IP web link',
      weight: 3,
    },
    {
      regex: /\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:xyz|top|click|tk|ml|ga|cf|gq|work|loan|vip)(?:\/[^\s]*)?/i,
      label: 'Uses an unfamiliar or high-risk domain extension',
      weight: 2,
    },
    {
      regex: /\b(guaranteed\s*return|double\s*your\s*money|work\s*from\s*home.*daily|part[\s-]?time\s*job.*whatsapp|crypto\s*profit|guaranteed\s*profit)\b/i,
      label: 'Promises unrealistic investment or work-from-home returns',
      weight: 3,
    },
    {
      regex: /https?:\/\/[^\s]+/i,
      label: 'Contains an external web link to verify carefully before opening',
      weight: 1,
    },
  ];

  for (const p of criticalPatterns) {
    if (p.regex.test(lower)) {
      indicators.push(p.label);
      score += p.weight;
    }
  }

  // Category-specific checks when inspecting URLs or unknown phone numbers directly
  if (normalizedType === 'url' && score === 0) {
    const looksLikeDomain = /(?:https?:\/\/|www\.|[a-z0-9-]+\.[a-z]{2,})/i.test(lower);
    if (looksLikeDomain) {
      const suspiciousKeywordsInUrl = /(kyc|verify|login|secure|bank|update|account|wallet|refund|claim|support)/i.test(lower);
      if (suspiciousKeywordsInUrl) {
        indicators.push('Web link uses account-verification or banking keywords — verify official domain spelling');
        score += 3;
      } else {
        indicators.push('External link submitted — confirm the domain belongs to the official organization');
        score += 2;
      }
    }
  }

  if (normalizedType === 'phone') {
    const compactPhone = content.replace(/[\s().-]+/g, '');
    const looksLikePhone = /^\+?\d{7,15}$/.test(compactPhone);
    if (looksLikePhone && score < 2) {
      indicators.push('Unknown phone number — caller identity cannot be verified from a number alone');
      score = Math.max(score, 2);
    }
  }

  const requiresPause =
    score >= 3 ||
    /\b(urgent|immediately|within \d+ (min|hour)|today only|otp|upi|transfer|pay)\b/i.test(lower);

  if (score >= 4) {
    return { riskLevel: 'RED', indicators, requiresPause: true };
  }
  if (score >= 2) {
    return { riskLevel: 'YELLOW', indicators, requiresPause };
  }
  return { riskLevel: 'GREEN', indicators, requiresPause: false };
}

function buildHeuristicScamResult(
  heuristic: { riskLevel: RiskLevel; indicators: string[]; requiresPause: boolean },
  inputType: string,
  source: 'heuristic_fallback' | 'heuristic_offline',
  fallbackReason?: AiServiceErrorCode
): ScamAnalysisResult {
  const typeName = inputType.trim().toUpperCase() || 'MESSAGE';

  const defaultRedActions = [
    'Do not click any links or call numbers listed inside this message.',
    'Never share OTPs, passwords, or bank/UPI PINs.',
    'Do not send money or install remote-access apps like AnyDesk or TeamViewer.',
    'Contact your bank or official institution directly using the phone number on the back of your card or passbook.',
    'Check with a trusted family member before taking any action.',
  ];

  const defaultYellowActions = [
    'Pause and verify who actually sent this before replying or tapping anything.',
    'Open your official bank or service app directly rather than using links or numbers from the message.',
    'Ask a trusted family member to review this together with you.',
  ];

  const defaultGreenActions = [
    'No obvious scam signs were detected, though automated checks cannot guarantee 100% safety.',
    'Still avoid sharing OTPs, PINs, or passwords with anyone.',
  ];

  const matchedSummary =
    heuristic.indicators.length > 0 ? heuristic.indicators.slice(0, 3).join('; ') : '';

  let explanation: string;
  if (heuristic.riskLevel === 'RED') {
    explanation = `Warning: This ${typeName} shows strong scam signs (${matchedSummary}). Real banks and official agencies do not threaten sudden account blocking or pressure you to click links or share secret codes.`;
  } else if (heuristic.riskLevel === 'YELLOW') {
    explanation = matchedSummary
      ? `Caution: This ${typeName} contains warning signs (${matchedSummary}). Please verify it directly through an official app, card helpline, or family member before acting.`
      : `Caution: This ${typeName} should be verified independently before you reply, click links, or share personal details.`;
  } else {
    explanation = `No obvious scam patterns were detected in this ${typeName}. Even when a check looks safe, automated checks cannot guarantee 100% safety, so always verify unexpected requests.`;
  }

  return {
    riskLevel: heuristic.riskLevel,
    explanation,
    indicators:
      heuristic.indicators.length > 0
        ? heuristic.indicators
        : ['No known high-risk scam patterns detected in this message'],
    recommendedActions:
      heuristic.riskLevel === 'RED'
        ? defaultRedActions
        : heuristic.riskLevel === 'YELLOW'
        ? defaultYellowActions
        : defaultGreenActions,
    requiresPause: heuristic.requiresPause,
    source,
    ...(fallbackReason ? { fallbackReason } : {}),
  };
}

export async function analyzeScamContent(params: {
  content: string;
  inputType: string;
  language: SupportedLanguage;
}): Promise<ScamAnalysisResult> {
  const heuristic = runDeterministicScamChecks(params.content, params.inputType);
  const status = getAiStatus();

  if (!status.configured) {
    return buildHeuristicScamResult(heuristic, params.inputType, 'heuristic_offline', 'MISSING_API_KEY');
  }

  const ai = getClient();
  const modelName = getConfiguredGeminiModel();
  const scamAiTimeoutMs = isVercelRuntime() ? 7500 : 12000;

  try {
    const response = await withAttemptTimeout(
      (signal) =>
        ai.models.generateContent({
          model: modelName,
          contents: `Analyze this ${params.inputType} for scam, phishing, UPI/bank fraud, impersonation, or social engineering risk:\n\n"""${params.content}"""`,
          config: {
            abortSignal: signal,
            systemInstruction: [
              'You are a scam-protection advisor for senior citizens.',
              languageInstruction(params.language),
              'Classify riskLevel as GREEN (no obvious risk found), YELLOW (be careful / verify), or RED (likely scam).',
              'Never claim 100% certainty. Even for GREEN, remind the user that safety checks cannot guarantee safety.',
              'For RED, include clear protective instructions (no links, no OTPs/PINs/passwords, no remote apps, verify via official numbers, ask family).',
            ].join('\n'),
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                riskLevel: { type: Type.STRING, enum: ['GREEN', 'YELLOW', 'RED'] },
                explanation: { type: Type.STRING },
                indicators: { type: Type.ARRAY, items: { type: Type.STRING } },
                recommendedActions: { type: Type.ARRAY, items: { type: Type.STRING } },
              },
              required: ['riskLevel', 'explanation', 'indicators', 'recommendedActions'],
            },
          },
        }),
      scamAiTimeoutMs
    );

    const rawText = extractCandidateText(
      response as unknown as {
        text?: string;
        promptFeedback?: { blockReason?: string };
        candidates?: Array<{ finishReason?: string; content?: { parts?: Array<{ text?: string }> } }>;
      }
    );
    const fenceMatch = rawText.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
    const cleanJson = (fenceMatch ? fenceMatch[1] : rawText).trim();
    const parsed = JSON.parse(cleanJson) as {
      riskLevel?: RiskLevel;
      explanation?: string;
      indicators?: string[];
      recommendedActions?: string[];
    };

    // Never downgrade a deterministic finding below what rules already caught
    const rank: Record<RiskLevel, number> = { GREEN: 1, YELLOW: 2, RED: 3 };
    const aiLevel: RiskLevel = parsed.riskLevel && rank[parsed.riskLevel] ? parsed.riskLevel : heuristic.riskLevel;
    const finalLevel: RiskLevel = rank[heuristic.riskLevel] > rank[aiLevel] ? heuristic.riskLevel : aiLevel;

    const mergedIndicators = Array.from(
      new Set([
        ...(Array.isArray(parsed.indicators) ? parsed.indicators.filter((x) => typeof x === 'string' && x.trim()) : []),
        ...heuristic.indicators,
      ])
    ).slice(0, 6);

    const fallbackActions = buildHeuristicScamResult(
      { ...heuristic, riskLevel: finalLevel },
      params.inputType,
      'heuristic_fallback'
    ).recommendedActions;

    const cleanAiActions = Array.isArray(parsed.recommendedActions)
      ? parsed.recommendedActions.filter((x) => typeof x === 'string' && x.trim())
      : [];

    return {
      riskLevel: finalLevel,
      explanation:
        typeof parsed.explanation === 'string' && parsed.explanation.trim()
          ? parsed.explanation.trim()
          : buildHeuristicScamResult({ ...heuristic, riskLevel: finalLevel }, params.inputType, 'heuristic_fallback')
              .explanation,
      indicators:
        mergedIndicators.length > 0
          ? mergedIndicators
          : ['No immediate warning indicators detected, stay cautious'],
      recommendedActions: cleanAiActions.length > 0 ? cleanAiActions : fallbackActions,
      requiresPause: heuristic.requiresPause || finalLevel !== 'GREEN',
      source: 'ai',
    };
  } catch (err) {
    const classified = classifyGeminiError(err, modelName);
    console.warn('[Buddy Scam Check Fallback]', {
      endpoint: '/api/scam/analyze',
      rateLimitSource: classified.rateLimitSource ?? null,
      httpStatus: classified.statusCode,
      geminiErrorCode: classified.geminiErrorCode ?? classified.code,
      model: modelName,
      retryAfter: classified.retryAfterSeconds ?? null,
      retryCount: classified.retryCount ?? 0,
      inputType: params.inputType,
      fallbackReason: classified.code,
      timestamp: new Date().toISOString(),
    });
    return buildHeuristicScamResult(heuristic, params.inputType, 'heuristic_fallback', classified.code);
  }
}
