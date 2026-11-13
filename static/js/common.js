/* Shared helpers */
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function fmtMoney(n) {
  if (n == null || isNaN(n)) return '—';
  return (window.CV.currency || '') + new Intl.NumberFormat(window.CV.locale || 'en-IN', { maximumFractionDigits: 0 }).format(Math.round(n));
}
function fmtDate(iso) {
  const d = new Date(iso);
  return isNaN(d) ? '' : d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}
/* Always resolves to {ok:boolean, error?:string, ...}; never throws. */
async function CV_API(url, opts) {
  try {
    const res = await fetch(url, Object.assign({ headers: { 'Accept': 'application/json' } }, opts || {}));
    const data = await res.json().catch(() => null);
    if (!data) return { ok: false, error: 'Something went wrong. Please try again.' };
    return data;
  } catch (e) {
    return { ok: false, error: "We couldn't connect. Please check your internet connection and try again." };
  }
}
document.addEventListener('DOMContentLoaded', () => {
  const t = document.querySelector('.nav-toggle'), n = document.getElementById('nav');
  if (t && n) t.addEventListener('click', () => { const o = n.classList.toggle('open'); t.setAttribute('aria-expanded', o); });
  if (t && n) {
    t.addEventListener('click', () => {
      const o = n.classList.toggle('open');
      t.setAttribute('aria-expanded', o);
    });
    n.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        n.classList.remove('open');
        t.setAttribute('aria-expanded', 'false');
      });
    });
  }
});
