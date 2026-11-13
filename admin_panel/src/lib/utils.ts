import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Combines conditional Tailwind class names without conflicting utilities. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Formats a millisecond duration for compact dashboard display. */
export function formatMilliseconds(value: number): string {
  return `${Math.round(value).toLocaleString()} ms`;
}

/** Formats an ISO timestamp in the browser locale. */
export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}
