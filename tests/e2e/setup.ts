import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import type { FullConfig } from '@playwright/test';

export default async function setup(config: FullConfig): Promise<() => Promise<void>> {
  const baseURL = config.projects[0].use.baseURL!;
  let occupied = false;
  try {
    await fetch(`${baseURL}/tasks`, { signal: AbortSignal.timeout(1000) });
    occupied = true;
  } catch {}
  if (occupied) throw new Error(`E2E port is already in use: ${baseURL}. Stop that server first.`);

  const directory = await mkdtemp(join(tmpdir(), 'daily-planner-e2e-'));
  const child = spawn(process.env.PLAYWRIGHT_PYTHON || 'python', [
    '-m', 'uvicorn', 'tests.e2e_app:create_e2e_app', '--factory',
    '--host', '127.0.0.1', '--port', new URL(baseURL).port,
  ], {
    cwd: config.rootDir.endsWith('e2e') ? join(config.rootDir, '../..') : process.cwd(),
    shell: false,
    env: {
      ...process.env, PLANNER_AGENT_PROVIDER: 'deterministic',
      PLANNER_E2E_DB_PATH: join(directory, 'tasks.sqlite3'), PYTHONUNBUFFERED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  let launchError: Error | undefined;
  child.on('error', (error) => { launchError = error; });
  child.stdout.on('data', (data: Buffer) => { output = (output + data.toString()).slice(-8000); });
  child.stderr.on('data', (data: Buffer) => { output = (output + data.toString()).slice(-8000); });

  const cleanup = async (): Promise<void> => {
    try {
      if (child.pid && child.exitCode === null && child.signalCode === null) {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('E2E server did not stop within 5 seconds')), 5000);
          child.once('exit', () => { clearTimeout(timer); resolve(); });
          child.kill('SIGTERM');
        });
      }
    } finally {
      await rm(directory, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
    }
  };

  try {
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      if (launchError) throw launchError;
      if (child.exitCode !== null) throw new Error(`E2E server exited: ${output}`);
      try {
        const response = await fetch(`${baseURL}/tasks`, { signal: AbortSignal.timeout(1000) });
        if (response.ok) return cleanup;
      } catch {}
      await delay(100);
    }
    throw new Error(`E2E server startup timed out: ${output}`);
  } catch (error) {
    await cleanup();
    throw error;
  }
}