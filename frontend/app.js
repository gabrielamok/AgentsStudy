const $ = (selector) => document.querySelector(selector);
const localDate = (date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
const state = { tasks: [], drafts: [], editingId: null };
$('#day').value = localDate(new Date());

function message(text, error = false) {
  $('#status').textContent = text;
  $('#status').classList.toggle('error', error);
  $('#status').hidden = !text;
}

async function api(path, method = 'GET', body) {
  const response = await fetch(path, {
    method, headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    let detail = data.detail || 'Request failed. Please try again.';
    if (Array.isArray(detail)) detail = detail.map((entry) => `${entry.loc.slice(1).join('.')}: ${entry.msg}`).join('; ');
    if (data.conflicts) detail += `: ${data.conflicts.map((task) => task.title).join(', ')}`;
    throw new Error(detail);
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function action(label, className, callback) {
  const button = element('button', className, label);
  button.type = 'button';
  button.addEventListener('click', async () => {
    button.disabled = true;
    try { await callback(); } catch (error) { message(error.message, true); }
    finally { button.disabled = false; }
  });
  return button;
}

function resetEditor() {
  state.editingId = null;
  $('#task-form').reset();
  $('#editor-title').textContent = 'Add a task';
  $('#save-task').textContent = 'Add task';
  $('#cancel-edit').hidden = true;
}

function taskRow(task, draft = false) {
  const row = element('article', 'task-row');
  const clock = (value) => new Date(value).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  const time = element('div', 'task-time', clock(task.start_time));
  time.append(element('span', '', clock(task.end_time)));
  const content = element('div', `task-content ${task.priority}`);
  content.append(element('h3', '', task.title));
  const meta = element('div', 'task-meta');
  meta.append(element('span', '', `${task.priority} priority`), element('span', '', task.type));
  const actions = element('div', 'task-actions');
  if (draft) {
    actions.append(action('Add to schedule', '', async () => {
      const saved = await api('/tasks', 'POST', task);
      state.tasks.push(saved);
      state.drafts = state.drafts.filter((item) => item !== task);
      render();
      message(`Added "${task.title}".`);
    }), action('Discard', 'quiet', () => {
      state.drafts = state.drafts.filter((item) => item !== task);
      render();
    }));
  } else {
    actions.append(action('Edit', '', () => {
      state.editingId = task.id;
      $('#title').value = task.title;
      $('#start').value = task.start_time.slice(11, 16);
      $('#end').value = task.end_time.slice(11, 16);
      $('#priority').value = task.priority;
      $('#type').value = task.type;
      $('#editor-title').textContent = `Edit: ${task.title}`;
      $('#save-task').textContent = 'Save changes';
      $('#cancel-edit').hidden = false;
      $('#title').focus();
    }), action('Delete', 'delete', async () => {
      if (!confirm(`Delete "${task.title}" (${clock(task.start_time)} to ${clock(task.end_time)})?`)) return;
      await api(`/tasks/${task.id}`, 'DELETE');
      state.tasks = state.tasks.filter((item) => item.id !== task.id);
      if (state.editingId === task.id) resetEditor();
      render();
      message(`Deleted "${task.title}".`);
    }));
  }
  content.append(meta, actions);
  row.append(time, content);
  return row;
}

function render() {
  const selected = $('#day').value;
  const day = new Date(`${selected}T12:00:00`);
  $('#day-title').textContent = day.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
  const tasks = state.tasks.filter((task) => task.start_time.slice(0, 10) === selected).sort((first, second) => first.start_time.localeCompare(second.start_time));
  $('#task-count').textContent = `${tasks.length} ${tasks.length === 1 ? 'task' : 'tasks'}`;
  $('#task-list').replaceChildren(...tasks.map((task) => taskRow(task)));
  if (!tasks.length) {
    const empty = element('div', 'empty');
    empty.append(element('strong', '', 'A little breathing room.'), element('p', '', 'No tasks scheduled.'));
    $('#task-list').append(empty);
  }
  $('#draft-list').replaceChildren(...state.drafts.map((task) => taskRow(task, true)));
}

$('#cancel-edit').addEventListener('click', resetEditor);
$('#day').addEventListener('change', () => {
  if (!$('#day').value) return;
  resetEditor(); state.drafts = [];
  $('#draft-section').hidden = true;
  message(''); render();
});

$('#task-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#save-task');
  button.disabled = true;
  const task = {
    title: $('#title').value.trim(), start_time: `${$('#day').value}T${$('#start').value}:00`,
    end_time: `${$('#day').value}T${$('#end').value}:00`, priority: $('#priority').value, type: $('#type').value,
  };
  try {
    const editingId = state.editingId;
    const saved = await api(editingId ? `/tasks/${editingId}` : '/tasks', editingId ? 'PUT' : 'POST', task);
    state.tasks = state.tasks.filter((item) => item.id !== saved.id);
    state.tasks.push(saved);
    resetEditor(); render(); message(`Saved "${saved.title}".`);
  } catch (error) { message(error.message, true); }
  finally { button.disabled = false; }
});

$('#planner-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  button.textContent = 'Planning...';
  const selected = $('#day').value;
  try {
    const result = await api('/planner', 'POST', {
      request: $('#request').value, date: selected,
      work_start: $('#work-start').value, work_end: $('#work-end').value, buffer_minutes: Number($('#buffer').value),
    });
    if ($('#day').value !== selected) return;
    state.drafts = result.tasks;
    $('#warnings').replaceChildren(...result.warnings.map((warning) => element('li', '', warning)));
    $('#draft-section').hidden = false;
    render();
    message(`${result.tasks.length} proposed ${result.tasks.length === 1 ? 'task' : 'tasks'}. No saved tasks changed.`);
  } catch (error) { message(error.message, true); }
  finally { button.disabled = false; button.textContent = 'Create proposal'; }
});

async function initialize() {
  render();
  try { state.tasks = await api('/tasks'); render(); }
  catch (error) { message(`Could not load tasks. ${error.message}`, true); }
}
initialize();