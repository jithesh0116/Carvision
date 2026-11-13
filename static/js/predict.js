/* CarVision: 3-panel guided flow (photo -> select car -> details). Result is a separate page. */
(function () {
  const $ = id => document.getElementById(id);
  const S = { status: null, fields: [], imageName: null, detection: null, locked: false,
              variantVisible: false, touched: {}, opts: null, uploading: false };
  const stepEls = document.querySelectorAll('.stepper .step');
  const OK_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  const STATE_CITY_MAP = {
    'Andhra Pradesh': ['Guntur', 'Vijayawada', 'Visakhapatnam'],
    'Assam': ['Guwahati'],
    'Bihar': ['Patna'],
    'Chandigarh (UT)': ['Chandigarh'],
    'Chhattisgarh': ['Raipur'],
    'Delhi (NCT)': ['Delhi'],
    'Goa': ['Panaji'],
    'Gujarat': ['Ahmedabad', 'Bhavnagar', 'Gandhinagar', 'Rajkot', 'Surat', 'Vadodara'],
    'Haryana': ['Faridabad', 'Gurgaon', 'Panchkula'],
    'Jammu & Kashmir': ['Jammu', 'Srinagar'],
    'Jharkhand': ['Dhanbad', 'Jamshedpur', 'Ranchi'],
    'Karnataka': [
      'Bagalkote', 'Ballari', 'Belagavi', 'Bengaluru', 'Bhadravati', 'Bidar',
      'Chamarajanagar', 'Chikkaballapura', 'Chikkamagaluru', 'Chitradurga',
      'Davanagere', 'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Hosapete',
      'Hubballi', 'Kalaburagi', 'Karwar', 'Kolar', 'Koppal', 'Madikeri',
      'Mandya', 'Mangalore', 'Mangaluru', 'Mysuru', 'Raichur', 'Ramanagara',
      'Shivamogga', 'Tumakuru', 'Udupi', 'Vijayapura', 'Yadgir'
    ],
    'Kerala': ['Kochi', 'Kozhikode', 'Thiruvananthapuram', 'Thrissur'],
    'Madhya Pradesh': ['Bhopal', 'Gwalior', 'Indore', 'Jabalpur'],
    'Maharashtra': ['Aurangabad', 'Kolhapur', 'Mumbai', 'Nagpur', 'Nashik', 'Navi Mumbai', 'Pune', 'Solapur', 'Thane'],
    'Odisha': ['Bhubaneswar', 'Cuttack'],
    'Punjab': ['Amritsar', 'Jalandhar', 'Ludhiana', 'Mohali'],
    'Rajasthan': ['Jaipur', 'Jodhpur', 'Kota', 'Udaipur'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli'],
    'Telangana': ['Hyderabad', 'Warangal'],
    'Uttar Pradesh': ['Agra', 'Ghaziabad', 'Kanpur', 'Lucknow', 'Meerut', 'Noida', 'Prayagraj', 'Varanasi'],
    'Uttarakhand': ['Dehradun'],
    'West Bengal': ['Kolkata', 'Siliguri']
  };

  const CITY_TO_STATE = {};
  Object.entries(STATE_CITY_MAP).forEach(([st, cities]) => {
    cities.forEach(c => { CITY_TO_STATE[c.toLowerCase()] = st; });
  });

  const DEF = {
    km_driven: { label: 'Kilometres Driven', type: 'number', unit: 'km', step: 1, missing: 'Please enter the kilometres driven.', neg: 'Kilometres driven cannot be negative.', bad: 'Please enter a valid number of kilometres.' },
    fuel_type: { label: 'Fuel Type', type: 'select', missing: 'Please select the fuel type.' },
    transmission: { label: 'Transmission', type: 'select', missing: 'Please select the transmission.' },
    owner_count: { label: 'Number of Previous Owners', type: 'select', missing: 'Please select the number of previous owners.' },
    engine_cc: { label: 'Engine', type: 'number', unit: 'cc', step: 1, missing: 'Please enter the engine capacity.', neg: 'Engine capacity cannot be negative.', bad: 'Please enter a valid engine capacity.' },
    mileage: { label: 'Mileage', type: 'number', unit: 'km/l', step: 0.1, missing: 'Please enter the mileage.', neg: 'Mileage cannot be negative.', bad: 'Please enter a valid mileage.' },
    location: { label: 'City / Location', type: 'select', missing: 'Please select your city.' },
    vehicle_condition: { label: 'Vehicle Condition', type: 'select', missing: 'Please select the vehicle condition.' },
  };
  const ORDER = ['km_driven', 'fuel_type', 'transmission', 'owner_count', 'engine_cc', 'mileage', 'location', 'vehicle_condition'];
  const COND_HELP = { 'excellent': 'Almost like new', 'good': 'Well maintained with normal wear', 'average': 'Visible signs of regular use', 'needs repair': 'Requires noticeable repairs' };

  const sel = { brand: $('sel-brand'), model: $('sel-model'), variant: $('sel-variant'), year: $('sel-year') };
  const setErr = (name, msg) => { const e = $('err-' + name); if (e) e.textContent = msg || ''; };

  function go(n) {
    [1, 2, 3].forEach(i => $('panel-' + i).hidden = i !== n);
    stepEls.forEach((el, i) => {
      const k = i + 1;
      el.classList.toggle('active', k === n); el.classList.toggle('done', k < n);
      el.querySelector('.step-dot').textContent = k < n ? '✓' : k;
      k === n ? el.setAttribute('aria-current', 'step') : el.removeAttribute('aria-current');
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function banner(msg) { const b = $('banner'); b.textContent = msg; b.hidden = !msg; }

  /* ---------------------------------------------------------- Step 1: photo */
  function validateFile(f) {
    if (!OK_TYPES.includes(f.type)) return 'Please upload a JPG, PNG, or WEBP image.';
    if (f.size > S.status.max_upload_mb * 1024 * 1024) return 'That photo is too large. Please upload an image under ' + Math.floor(S.status.max_upload_mb) + ' MB.';
    return '';
  }
  function resetPhoto() {
    S.imageName = null; S.detection = null;
    $('preview').hidden = true; $('dropzone').hidden = false; $('detect').hidden = true;
    $('file-input').value = ''; $('next-1').disabled = true; setErr('image', ''); $('upload-status').textContent = '';
  }
  function pick(file) {
    if (S.locked || !file) return;
    setErr('image', '');
    const problem = validateFile(file);
    if (problem) { setErr('image', problem); return; }
    const url = URL.createObjectURL(file), img = $('preview-img');
    img.onerror = () => { URL.revokeObjectURL(url); resetPhoto(); setErr('image', "We couldn't read this image. Please try another photo."); };
    img.onload = () => upload(file);
    img.src = url;
    $('dropzone').hidden = true; $('preview').hidden = false; $('detect').hidden = true; $('next-1').disabled = true;
    $('upload-status').textContent = 'Checking your photo…';
  }
  async function upload(file) {
    S.uploading = true;
    const fd = new FormData(); fd.append('image', file);
    const r = await CV_API('/api/upload-image', { method: 'POST', body: fd });
    S.uploading = false;
    if (!r.ok) { resetPhoto(); setErr('image', r.error); return; }
    S.imageName = r.filename; S.detection = r.detection;
    $('upload-status').textContent = 'Photo ready.';
    $('next-1').disabled = false;
    const d = r.detection;
    if (d && d.confident && d.brand && d.model) {
      $('detect-label').textContent = d.brand + ' ' + d.model; $('detect').hidden = false;
    }
  }
  $('btn-upload').addEventListener('click', e => { e.stopPropagation(); $('file-input').click(); });
  $('dropzone').addEventListener('click', () => $('file-input').click());
  $('dropzone').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); } });
  ['dragenter', 'dragover'].forEach(ev => $('dropzone').addEventListener(ev, e => { e.preventDefault(); $('dropzone').classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev => $('dropzone').addEventListener(ev, e => { e.preventDefault(); $('dropzone').classList.remove('drag'); }));
  $('dropzone').addEventListener('drop', e => pick(e.dataTransfer.files[0]));
  $('file-input').addEventListener('change', e => pick(e.target.files[0]));
  $('btn-change').addEventListener('click', () => { resetPhoto(); $('file-input').click(); });
  $('btn-remove').addEventListener('click', resetPhoto);
  $('next-1').addEventListener('click', () => go(2));
  $('skip-1').addEventListener('click', () => go(2));
  $('detect-manual').addEventListener('click', () => { $('detect').hidden = true; });
  $('detect-use').addEventListener('click', async () => {
    const d = S.detection; $('detect').hidden = true; go(2);
    sel.brand.value = d.brand; await onBrand(); sel.model.value = d.model; await onModel();
  });

  /* ---------------------------------------------------------- Step 2: brand -> model -> variant -> year */
  function fill(el, items, placeholder) {
    el.innerHTML = '<option value="">' + esc(placeholder) + '</option>' + items.map(v => '<option value="' + esc(v) + '">' + esc(v) + '</option>').join('');
    el.disabled = !items.length;
  }
  function reset(...names) { const ph = { model: 'Select Model', variant: 'Select Variant', year: 'Select Year' }; names.forEach(n => { fill(sel[n], [], ph[n]); setErr(n, ''); }); }
  function updateNext2() {
    $('next-2').disabled = !(sel.brand.value && sel.model.value && (!S.variantVisible || sel.variant.value) && sel.year.value);
  }
  async function loadBrands() {
    const r = await CV_API('/api/brands');
    if (!r.ok) { setErr('brand', r.error); return; }
    fill(sel.brand, r.brands, 'Select Brand');
  }
  async function onBrand() {
    reset('model', 'variant', 'year'); S.variantVisible = false; updateNext2();
    if (!sel.brand.value) return;
    sel.model.innerHTML = '<option value="">Loading…</option>';
    const r = await CV_API('/api/models?brand=' + encodeURIComponent(sel.brand.value));
    if (!r.ok) { reset('model'); setErr('model', r.error); return; }
    fill(sel.model, r.models, 'Select Model');
  }
  async function onModel() {
    reset('variant', 'year'); S.variantVisible = false; updateNext2();
    if (!sel.model.value) return;
    const base = '?brand=' + encodeURIComponent(sel.brand.value) + '&model=' + encodeURIComponent(sel.model.value);
    if (S.fields.includes('variant')) {
      const r = await CV_API('/api/variants' + base);
      if (!r.ok) { setErr('variant', r.error); return; }
      S.variantVisible = r.variants.length > 0;
      $('wrap-variant').hidden = !S.variantVisible;
      if (S.variantVisible) { fill(sel.variant, r.variants, 'Select Variant'); return; }
    }
    await loadYears();
  }
  async function loadYears() {
    let q = '?brand=' + encodeURIComponent(sel.brand.value) + '&model=' + encodeURIComponent(sel.model.value);
    if (S.variantVisible && sel.variant.value) q += '&variant=' + encodeURIComponent(sel.variant.value);
    sel.year.innerHTML = '<option value="">Loading…</option>';
    const r = await CV_API('/api/years' + q);
    if (!r.ok) { reset('year'); setErr('year', r.error); return; }
    fill(sel.year, r.years.map(String), 'Select Year');
    if (r.years.length === 1) { sel.year.value = String(r.years[0]); }
    updateNext2();
  }
  sel.brand.addEventListener('change', onBrand);
  sel.model.addEventListener('change', onModel);
  sel.variant.addEventListener('change', () => { reset('year'); updateNext2(); if (sel.variant.value) loadYears(); });
  sel.year.addEventListener('change', updateNext2);
  $('back-2').addEventListener('click', () => go(1));
  $('next-2').addEventListener('click', async () => { $('next-2').disabled = true; await buildDetails(); go(3); updateNext2(); });

  /* ---------------------------------------------------------- Step 3: details */
  const F = name => $('f-' + name);
  function vehicleQuery(extra) {
    const p = new URLSearchParams({ brand: sel.brand.value, model: sel.model.value });
    if (S.variantVisible && sel.variant.value) p.set('variant', sel.variant.value);
    Object.entries(extra || {}).forEach(([k, v]) => v && p.set(k, v));
    return p.toString();
  }
  async function buildDetails() {
    S.touched = {}; setErr('form', '');
    $('vehicle-line').textContent = [sel.brand.value, sel.model.value, S.variantVisible ? sel.variant.value : ''].filter(Boolean).join(' ') + ' · ' + sel.year.value;
    const o = await CV_API('/api/options?' + vehicleQuery());
    if (!o.ok) { $('details-form').innerHTML = ''; setErr('form', o.error); return; }
    S.opts = o;
    const availableStates = Object.keys(STATE_CITY_MAP).sort((a, b) => a.localeCompare(b));

    const html = ORDER.filter(n => S.fields.includes(n)).map(n => {
      const d = DEF[n];
      if (n === 'location') {
        const stateControl = '<div class="field" id="field-state">' +
          '<label for="f-state">State / Union Territory</label>' +
          '<select id="f-state"><option value="">Select State</option>' +
          availableStates.map(s => '<option value="' + esc(s) + '">' + esc(s) + '</option>').join('') +
          '</select>' +
          '<p class="error" id="err-state" role="alert"></p></div>';

        const cityControl = '<div class="field" id="field-location">' +
          '<label for="f-location">City / Location</label>' +
          '<select id="f-location" data-name="location" disabled>' +
          '<option value="">Select State First</option>' +
          '</select>' +
          '<p class="error" id="err-location" role="alert"></p></div>';

        return stateControl + cityControl;
      }
      let control;
      if (d.type === 'select') {
        const items = (o.choices[n] || []);
        const single = items.length === 1;
        control = '<select id="f-' + n + '" data-name="' + n + '"><option value="">Select</option>' +
          items.map(v => '<option value="' + esc(v) + '"' + (single ? ' selected' : '') + '>' + esc(v) + '</option>').join('') + '</select>';
        if (n === 'vehicle_condition') {
          const lines = items.filter(v => COND_HELP[String(v).toLowerCase()]).map(v => '<li><b>' + esc(v) + '</b> — ' + esc(COND_HELP[String(v).toLowerCase()]) + '</li>');
          if (lines.length) control += '<ul class="help-list">' + lines.join('') + '</ul>';
        }
      } else {
        control = '<div class="input-unit"><input id="f-' + n + '" data-name="' + n + '" type="number" inputmode="decimal" min="0" step="' + d.step + '"><span>' + d.unit + '</span></div>' +
          '<p class="hint" id="hint-' + n + '"></p>';
      }
      return '<div class="field" id="field-' + n + '"><label for="f-' + n + '">' + d.label + '</label>' + control + '<p class="error" id="err-' + n + '" role="alert"></p></div>';
    }).join('');
    $('details-form').innerHTML = html;

    const stateEl = $('f-state');
    const locEl = $('f-location');
    if (stateEl && locEl) {
      stateEl.addEventListener('change', () => {
        const selectedState = stateEl.value;
        setErr('state', '');
        if (!selectedState) {
          locEl.innerHTML = '<option value="">Select State First</option>';
          locEl.disabled = true;
          locEl.value = '';
        } else {
          const stateCities = STATE_CITY_MAP[selectedState] || [];
          const cities = stateCities.slice().sort((a, b) => a.localeCompare(b));

          locEl.innerHTML = '<option value="">Select City</option>' +
            cities.map(c => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
          locEl.disabled = false;
          if (cities.length === 1) {
            locEl.value = cities[0];
          }
        }
        S.touched['location'] = true;
        validate(false);
      });
    }

    $('details-form').querySelectorAll('[data-name]').forEach(el => {
      el.addEventListener('input', () => { S.touched[el.dataset.name] = true; validate(false); });
      el.addEventListener('change', () => {
        S.touched[el.dataset.name] = true; validate(false);
        if (el.dataset.name === 'fuel_type' || el.dataset.name === 'transmission') refreshSuggestions();
      });
      el.addEventListener('blur', () => validateOne(el.dataset.name, true));
    });
    applySuggestions(o.suggestions);
    validate(false);
  }
  function applySuggestions(s) {
    Object.entries(s || {}).forEach(([n, v]) => {
      const input = F(n), hint = $('hint-' + n);
      if (!input || !hint) return;
      if (!S.touched[n]) input.value = v.value;
      hint.textContent = 'Typical for similar cars in our data: ' + v.value + ' ' + DEF[n].unit + ' (' + v.n + ' listing' + (v.n === 1 ? '' : 's') + '). Please make sure it matches your car.';
    });
  }
  async function refreshSuggestions() {
    const extra = { fuel_type: F('fuel_type') ? F('fuel_type').value : '', transmission: F('transmission') ? F('transmission').value : '' };
    const o = await CV_API('/api/options?' + vehicleQuery(extra));
    if (o.ok) applySuggestions(o.suggestions);
    validate(false);
  }
  function fieldError(n) {
    const d = DEF[n], el = F(n); if (!el) return '';
    const raw = String(el.value).trim();
    if (raw === '') {
      if (n === 'location') {
        const stateEl = $('f-state');
        if (stateEl && !stateEl.value) {
          return 'Please select your state and city.';
        }
        return 'Please select your city.';
      }
      return d.missing;
    }
    if (d.type === 'number') {
      const v = Number(raw);
      if (!isFinite(v)) return d.bad;
      if (v < 0) return d.neg;
      if (v === 0 && n !== 'km_driven') return d.missing;
    }
    return '';
  }
  function validateOne(n, show) {
    const msg = fieldError(n);
    if (show || S.touched[n]) { setErr(n, msg); const f = $('field-' + n); if (f) f.classList.toggle('invalid', !!msg); }
    return !msg;
  }
  function validate(showAll) {
    let ok = true;
    ORDER.filter(n => S.fields.includes(n)).forEach(n => { if (!validateOne(n, showAll)) ok = false; });
    $('btn-predict').disabled = !(ok && S.status.model_ready);
    return ok;
  }
  $('back-3').addEventListener('click', () => go(2));
  $('btn-predict').addEventListener('click', async () => {
    if (!validate(true)) return;
    setErr('form', '');
    const payload = { brand: sel.brand.value, model: sel.model.value, year: sel.year.value, image: S.imageName };
    if (S.variantVisible) payload.variant = sel.variant.value;
    ORDER.filter(n => S.fields.includes(n)).forEach(n => { payload[n] = F(n).value; });
    $('loading').hidden = false;
    const r = await CV_API('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(payload) });
    if (r.ok) { location.href = r.redirect; return; }
    $('loading').hidden = true;
    const errs = r.errors || (r.field ? { [r.field]: r.error } : {});
    Object.entries(errs).forEach(([n, m]) => { if ($('err-' + n)) setErr(n, m); });
    setErr('form', r.error);
    if (['brand', 'model', 'variant', 'year'].includes(r.field)) go(2);
    else if (r.field === 'image') { go(1); setErr('image', r.error); }
  });

  /* ---------------------------------------------------------- init */
  async function init() {
    const st = await CV_API('/api/status');
    if (!st.ok) { S.locked = true; banner(st.error); $('btn-upload').disabled = true; return; }
    S.status = st; S.fields = st.input_fields || [];
    $('max-mb').textContent = Math.floor(st.max_upload_mb);
    if (!st.require_image) $('skip-1').hidden = false;
    if (!st.vehicles) {
      S.locked = true; $('btn-upload').disabled = true;
      banner('Vehicle data has not been loaded yet. Please ask the administrator to import the dataset.');
      return;
    }
    if (!st.model_ready) banner('Price estimates are not available yet because the price model has not been trained. You can look around, but the final estimate is unavailable for now.');
    $('wrap-variant').hidden = true;
    await loadBrands();
  }
  init();
})();
