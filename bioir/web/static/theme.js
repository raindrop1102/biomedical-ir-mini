  const themeToggle = document.getElementById('theme-toggle');
  function syncThemeButton() {
    const dark = document.documentElement.dataset.theme === 'dark';
    themeToggle.setAttribute('aria-pressed', String(dark));
    themeToggle.textContent = dark ? '☀ 淺色模式' : '☾ 深色模式';
    themeToggle.hidden = false;
  }
  themeToggle.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('biomedicalir-theme', theme); } catch (_) {}
    syncThemeButton();
  });
  syncThemeButton();
