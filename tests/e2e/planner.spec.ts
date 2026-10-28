import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

const TODAY = '2026-10-27';

type TaskInput = {
  title: string;
  start_time: string;
  end_time: string;
  priority: 'high' | 'medium' | 'low';
  type: 'fixed' | 'flexible';
};
type Task = TaskInput & { id: number };
type Plan = { date: string; tasks: TaskInput[]; warnings: string[] };

async function seedTask(request: APIRequestContext, changes: Partial<TaskInput> = {}): Promise<Task> {
  const response = await request.post('/tasks', {
    data: {
      title: 'Review pull request', start_time: `${TODAY}T09:00:00`,
      end_time: `${TODAY}T10:00:00`, priority: 'high', type: 'flexible', ...changes,
    },
  });
  expect(response.status()).toBe(201);
  return response.json();
}

async function savedTasks(request: APIRequestContext): Promise<Task[]> {
  const response = await request.get('/tasks');
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function openAgenda(page: Page): Promise<void> {
  const loaded = page.waitForResponse((response) => response.url().endsWith('/tasks') && response.request().method() === 'GET');
  await page.goto('/');
  await loaded;
  await expect(page.getByLabel('Selected day')).toHaveValue(TODAY);
}

test.beforeEach(async ({ page, request }) => {
  for (const task of await savedTasks(request)) {
    const response = await request.delete(`/tasks/${task.id}`);
    expect(response.status()).toBe(204);
  }
  await page.clock.install({ time: new Date(`${TODAY}T09:00:00Z`) });
  await openAgenda(page);
});

test('creating a task persists validated fields', async ({ page, request }) => {
  const input: TaskInput = {
    title: 'Study AI-103', start_time: `${TODAY}T09:00:00`, end_time: `${TODAY}T10:00:00`,
    priority: 'high', type: 'flexible',
  };
  await page.getByLabel('Title', { exact: true }).fill(input.title);
  await page.getByLabel('Start', { exact: true }).fill('09:00');
  await page.getByLabel('End', { exact: true }).fill('10:00');
  await page.getByRole('combobox', { name: 'Priority', exact: true }).selectOption(input.priority);
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption(input.type);
  const created = page.waitForResponse((response) => response.url().endsWith('/tasks') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Add task', exact: true }).click();
  const response = await created;
  expect(response.status()).toBe(201);
  const task: Task = await response.json();
  expect(task).toEqual({ ...input, id: expect.any(Number) });
  await expect(page.locator('#task-list h3')).toHaveText([input.title]);
  expect(await savedTasks(request)).toEqual([task]);
  await openAgenda(page);
  await expect(page.locator('#task-list h3')).toHaveText([input.title]);
});

test("listing today's tasks excludes other dates and orders by start time", async ({ page, request }) => {
  const lunch = await seedTask(request, { title: 'Lunch break', start_time: `${TODAY}T12:00:00`, end_time: `${TODAY}T13:00:00`, type: 'fixed' });
  const morning = await seedTask(request);
  await seedTask(request, { title: 'Yesterday only', start_time: '2026-10-26T09:00:00', end_time: '2026-10-26T10:00:00' });
  await seedTask(request, { title: 'Tomorrow only', start_time: '2026-10-28T09:00:00', end_time: '2026-10-28T10:00:00' });
  await openAgenda(page);
  await expect(page.locator('#task-list h3')).toHaveText([morning.title, lunch.title]);
  await expect(page.locator('#task-count')).toHaveText('2 tasks');
  expect(await savedTasks(request)).toHaveLength(4);
});

test('detecting overlapping tasks rejects the write and preserves the original', async ({ page, request }) => {
  const original = await seedTask(request);
  await openAgenda(page);
  await page.getByLabel('Title', { exact: true }).fill('Overlapping work');
  await page.getByLabel('Start', { exact: true }).fill('09:30');
  await page.getByLabel('End', { exact: true }).fill('10:30');
  const rejected = page.waitForResponse((response) => response.url().endsWith('/tasks') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Add task', exact: true }).click();
  const response = await rejected;
  expect(response.status()).toBe(409);
  expect(await response.json()).toEqual({ detail: expect.any(String), conflicts: [original] });
  await expect(page.locator('#status.error')).toBeVisible();
  await expect(page.locator('#task-list h3')).toHaveText([original.title]);
  expect(await savedTasks(request)).toEqual([original]);
});

test('updating a task persists new fields without creating a duplicate', async ({ page, request }) => {
  const original = await seedTask(request);
  await openAgenda(page);
  await page.getByRole('button', { name: 'Edit', exact: true }).click();
  await expect(page.getByLabel('Title', { exact: true })).toHaveValue(original.title);
  await page.getByLabel('Title', { exact: true }).fill('Updated review');
  await page.getByLabel('Start', { exact: true }).fill('10:00');
  await page.getByLabel('End', { exact: true }).fill('11:00');
  await page.getByRole('combobox', { name: 'Priority', exact: true }).selectOption('low');
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption('fixed');
  const updated = page.waitForResponse((response) => response.url().endsWith(`/tasks/${original.id}`) && response.request().method() === 'PUT');
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const response = await updated;
  expect(response.status()).toBe(200);
  const task: Task = await response.json();
  expect(task).toEqual({ ...original, title: 'Updated review', start_time: `${TODAY}T10:00:00`, end_time: `${TODAY}T11:00:00`, priority: 'low', type: 'fixed' });
  expect(await savedTasks(request)).toEqual([task]);
  await openAgenda(page);
  await expect(page.locator('#task-list h3')).toHaveText([task.title]);
});

test('deleting a task identifies it in the confirmation and removes it', async ({ page, request }) => {
  const task = await seedTask(request);
  await openAgenda(page);
  page.once('dialog', async (dialog) => {
    expect(dialog.type()).toBe('confirm');
    expect(dialog.message()).toContain(task.title);
    await dialog.accept();
  });
  const deleted = page.waitForResponse((response) => response.url().endsWith(`/tasks/${task.id}`) && response.request().method() === 'DELETE');
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  expect((await deleted).status()).toBe(204);
  await expect(page.locator('#task-list article')).toHaveCount(0);
  expect(await savedTasks(request)).toEqual([]);
  await openAgenda(page);
  await expect(page.locator('#task-count')).toHaveText('0 tasks');
});

test('submitting natural language returns structured non-overlapping proposals without autosaving', async ({ page, request }) => {
  const existing = await seedTask(request, { title: 'Protected appointment', type: 'fixed', end_time: `${TODAY}T09:30:00` });
  await openAgenda(page);
  await page.getByLabel('Planning request', { exact: true }).fill('Study AI-103 for 1 hour before 11 AM at high priority, and have lunch from noon to 1 PM.');
  const planned = page.waitForResponse((response) => response.url().endsWith('/planner') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Create proposal', exact: true }).click();
  const response = await planned;
  expect(response.status()).toBe(200);
  const plan: Plan = await response.json();
  expect(plan.date).toBe(TODAY);
  expect(plan.warnings).toEqual([]);
  expect(plan.tasks).toHaveLength(2);
  expect(plan.tasks.filter((task) => task.type === 'flexible'
    && task.priority === 'high' && task.end_time <= `${TODAY}T11:00:00`)).toHaveLength(1);
  const allTasks = [existing, ...plan.tasks];
  for (let first = 0; first < allTasks.length; first++) {
    for (let second = first + 1; second < allTasks.length; second++) {
      const overlap = allTasks[first].start_time < allTasks[second].end_time && allTasks[first].end_time > allTasks[second].start_time;
      expect(overlap).toBe(false);
    }
  }
  await expect(page.locator('#draft-list h3')).toHaveText(plan.tasks.map((task) => task.title));
  expect(await savedTasks(request)).toEqual([existing]);
  const accepted = page.waitForResponse((result) => result.url().endsWith('/tasks') && result.request().method() === 'POST');
  await page.getByRole('button', { name: 'Add to schedule', exact: true }).first().click();
  expect((await accepted).status()).toBe(201);
  await expect(page.locator('#task-list article')).toHaveCount(2);
  await expect(page.locator('#draft-list article')).toHaveCount(1);
  expect(await savedTasks(request)).toEqual([
    existing, { ...plan.tasks[0], id: expect.any(Number) },
  ]);
  await openAgenda(page);
  await expect(page.locator('#task-list h3')).toHaveText([existing.title, plan.tasks[0].title]);
});

test('invalid task request shows a recoverable error without changing saved tasks', async ({ page, request }) => {
  const original = await seedTask(request);
  await openAgenda(page);
  await page.getByLabel('Title', { exact: true }).fill('Invalid interval');
  await page.getByLabel('Start', { exact: true }).fill('12:00');
  await page.getByLabel('End', { exact: true }).fill('11:00');
  const rejected = page.waitForResponse((response) => response.url().endsWith('/tasks') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Add task', exact: true }).click();
  expect((await rejected).status()).toBe(422);
  await expect(page.locator('#status.error')).toBeVisible();
  await expect(page.locator('#status')).not.toBeEmpty();
  await expect(page.getByRole('button', { name: 'Add task', exact: true })).toBeEnabled();
  await expect(page.getByLabel('Title', { exact: true })).toHaveValue('Invalid interval');
  await expect(page.locator('#task-list h3')).toHaveText([original.title]);
  expect(await savedTasks(request)).toEqual([original]);
});

test('handling invalid AI output shows an error without returning or saving malformed tasks', async ({ page, request }) => {
  const original = await seedTask(request);
  await openAgenda(page);
  await page.getByLabel('Planning request', { exact: true }).fill('invalid-ai-output');
  const rejected = page.waitForResponse((response) => response.url().endsWith('/planner') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Create proposal', exact: true }).click();
  const response = await rejected;
  expect(response.status()).toBe(502);
  expect(await response.json()).toEqual({ detail: expect.any(String) });
  await expect(page.locator('#status.error')).toBeVisible();
  await expect(page.locator('#status')).not.toBeEmpty();
  await expect(page.locator('#status')).not.toContainText('private fixture diagnostic');
  await expect(page.locator('#draft-section')).toBeHidden();
  await expect(page.getByRole('button', { name: 'Create proposal', exact: true })).toBeEnabled();
  await expect(page.locator('#task-list h3')).toHaveText([original.title]);
  expect(await savedTasks(request)).toEqual([original]);
});