import fs from 'node:fs';
import path from 'node:path';
import dotenv from 'dotenv';

let envLoaded = false;

const PLACEHOLDER_VALUES = new Set([
  'MY_GEMINI_API_KEY',
  'MY_APP_URL',
  'your-gemini-api-key',
  'your-strong-admin-passcode',
  'your-messaging-number-with-country-code',
  '+<country_code><number>',
  'change-me',
]);

export function loadServerEnv(): void {
  if (envLoaded) return;
  envLoaded = true;

  const cwd = process.cwd();
  const envLocalPath = path.resolve(cwd, '.env.local');
  const envPath = path.resolve(cwd, '.env');

  // Existing process.env (host / AI Studio / Vercel) takes highest precedence,
  // followed by .env.local, then .env.
  if (fs.existsSync(envLocalPath)) {
    dotenv.config({ path: envLocalPath, override: false });
  }
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath, override: false });
  }
}

loadServerEnv();

export function isPlaceholderValue(value: string | undefined | null): boolean {
  if (!value) return true;
  const trimmed = value.trim();
  if (!trimmed) return true;
  return PLACEHOLDER_VALUES.has(trimmed);
}

export function getGeminiApiKey(): string | null {
  loadServerEnv();
  const raw = process.env.GEMINI_API_KEY;
  if (isPlaceholderValue(raw)) return null;
  return raw!.trim();
}

export function getAdminPasscode(): string | null {
  loadServerEnv();
  const raw = process.env.ADMIN_PASSCODE;
  if (isPlaceholderValue(raw)) return null;
  return raw!.trim();
}

export function normalizePhoneWithCountryCode(
  raw: string
): { valid: true; e164: string } | { valid: false; error: string } {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { valid: false, error: 'Phone number cannot be empty.' };
  }

  // Strip common formatting punctuation while preserving leading +
  const collapsed = trimmed.replace(/[\s().-]+/g, '');
  if (!collapsed.startsWith('+')) {
    return {
      valid: false,
      error: 'Phone number must include a country code starting with "+" (8 to 15 digits).',
    };
  }

  const digits = collapsed.slice(1);
  if (!/^[1-9]\d{7,14}$/.test(digits)) {
    return {
      valid: false,
      error: 'Phone number must start with "+" followed by country code and 8 to 15 digits total.',
    };
  }

  return { valid: true, e164: `+${digits}` };
}

export function getRawEnvMessagingNumber(): string | null {
  loadServerEnv();
  const raw = process.env.BUDDY_MESSAGING_NUMBER;
  if (isPlaceholderValue(raw)) return null;
  return raw!.trim();
}
