import os
import json


def radar_page_html() -> str:
    audiom_map_url = os.getenv("AUDIOM_MAP_URL", "https://www.audiom.net/map")
    audiom_api_key = os.getenv("AUDIOM_API_KEY", "")
    audiom_rules_path = os.getenv("AUDIOM_STORM_RULES_PATH", "/rules/arw-storms.json")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARW Radar Map</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    body {{
      margin: 0;
      background: Canvas;
      color: CanvasText;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 1.75rem;
      margin: 0 0 20px;
    }}
    form {{
      display: grid;
      gap: 16px;
      max-width: 680px;
    }}
    fieldset {{
      border: 1px solid ButtonBorder;
      border-radius: 6px;
      padding: 16px;
    }}
    legend {{
      font-weight: 700;
      padding: 0 6px;
    }}
    label {{
      display: block;
      font-weight: 650;
      margin-top: 10px;
    }}
    input {{
      box-sizing: border-box;
      display: block;
      width: 100%;
      max-width: 360px;
      margin-top: 4px;
      padding: 8px;
      font: inherit;
    }}
    .actions,
    .timeline-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 16px;
    }}
    button,
    .link-button {{
      min-height: 40px;
      padding: 8px 12px;
      border: 1px solid ButtonBorder;
      border-radius: 6px;
      background: ButtonFace;
      color: ButtonText;
      font: inherit;
      cursor: pointer;
    }}
    .link-button {{
      text-decoration: underline;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    #status {{
      margin-top: 18px;
      min-height: 1.5em;
      font-weight: 650;
    }}
    #map-panel {{
      margin-top: 24px;
    }}
    iframe {{
      width: 100%;
      height: min(76vh, 760px);
      border: 1px solid ButtonBorder;
      border-radius: 6px;
      background: Canvas;
    }}
    .summary {{
      margin: 0 0 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>ARW Radar Map</h1>

    <section id="search-panel" aria-labelledby="search-heading">
      <h2 id="search-heading">Search for a radar scan</h2>
      <form id="search-form">
        <fieldset>
          <legend>Location</legend>
          <label for="zipcode">ZIP code</label>
          <input id="zipcode" name="zipcode" inputmode="numeric" autocomplete="postal-code">

          <p>Or enter a city and state.</p>

          <label for="city">City</label>
          <input id="city" name="city" autocomplete="address-level2">

          <label for="state">State</label>
          <input id="state" name="state" autocomplete="address-level1" maxlength="32">
        </fieldset>

        <fieldset>
          <legend>Date and time</legend>
          <label for="datetime">Scan time</label>
          <input id="datetime" name="datetime" type="datetime-local">
        </fieldset>

        <div class="actions">
          <button type="submit">Load radar map</button>
        </div>
      </form>
    </section>

    <div id="status" role="status" aria-live="polite"></div>

    <section id="map-panel" hidden aria-labelledby="map-heading">
      <h2 id="map-heading">Radar scan map</h2>
      <p id="scan-summary" class="summary"></p>
      <div class="timeline-actions">
        <button id="search-again" type="button" class="link-button">Search again</button>
        <button id="prev-12" type="button">Previous 12 hours</button>
        <button id="prev-6" type="button">Previous 6 hours</button>
        <button id="prev-1" type="button">Previous hour</button>
        <button id="next-1" type="button">Next hour</button>
        <button id="next-6" type="button">Next 6 hours</button>
        <button id="next-12" type="button">Next 12 hours</button>
      </div>
      <iframe id="audiom-frame" title="Audiom radar scan map" allow="autoplay"></iframe>
    </section>
  </main>

  <script>
    const AUDIOM_MAP_URL = {json.dumps(audiom_map_url)};
    const AUDIOM_API_KEY = {json.dumps(audiom_api_key)};
    const AUDIOM_RULES_PATH = {json.dumps(audiom_rules_path)};
    const form = document.getElementById('search-form');
    const searchPanel = document.getElementById('search-panel');
    const mapPanel = document.getElementById('map-panel');
    const statusEl = document.getElementById('status');
    const frame = document.getElementById('audiom-frame');
    const scanSummary = document.getElementById('scan-summary');
    const searchAgain = document.getElementById('search-again');
    const navButtons = {{
      prev12: document.getElementById('prev-12'),
      prev6: document.getElementById('prev-6'),
      prev1: document.getElementById('prev-1'),
      next1: document.getElementById('next-1'),
      next6: document.getElementById('next-6'),
      next12: document.getElementById('next-12'),
    }};

    let currentSearch = null;
    let currentScanTime = null;
    let currentLocation = null;

    function setStatus(message) {{
      statusEl.textContent = message;
    }}

    function normalizeDate(value) {{
      if (!value) return null;
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return null;
      return parsed;
    }}

    function toApiDatetime(date) {{
      return date.toISOString().replace('.000Z', '+00:00');
    }}

    function appendLocationParams(params, search) {{
      if (search.zipcode) {{
        params.set('zipcode', search.zipcode);
      }} else {{
        params.set('city', search.city);
        params.set('state', search.state);
      }}
    }}

    function buildStormApiUrl(search, scanTime) {{
      const params = new URLSearchParams();
      appendLocationParams(params, search);
      if (scanTime) params.set('datetime', toApiDatetime(scanTime));
      return `/map/storms?${{params.toString()}}`;
    }}

    function buildStormGeoJsonUrl(search, scanTime) {{
      const params = new URLSearchParams();
      appendLocationParams(params, search);
      params.set('mode', 'audiom');
      if (scanTime) params.set('datetime', toApiDatetime(scanTime));
      return `${{window.location.origin}}/map/storms.geojson?${{params.toString()}}`;
    }}

    function buildAudiomUrl(search, scanTime, location) {{
      const stormUrl = buildStormGeoJsonUrl(search, scanTime);
      const params = new URLSearchParams();
      params.set('sources', 'storms');
      params.set('storms.type', 'geojson');
      params.set('storms.url', stormUrl);
      params.set('storms.rules', AUDIOM_RULES_PATH);
      params.set('storms.name', 'ARW Radar');
      params.set('center', `${{location.longitude}},${{location.latitude}}`);
      params.set('zoom', '7');
      params.set('stepsize', '1mi');
      if (AUDIOM_API_KEY) params.set('apiKey', AUDIOM_API_KEY);
      return `${{AUDIOM_MAP_URL}}?${{params.toString()}}`;
    }}

    function readSearchFromForm() {{
      const data = new FormData(form);
      const zipcode = String(data.get('zipcode') || '').trim();
      const city = String(data.get('city') || '').trim();
      const state = String(data.get('state') || '').trim();
      const datetimeValue = String(data.get('datetime') || '').trim();
      if (!zipcode && (!city || !state)) {{
        throw new Error('Enter either a ZIP code or both city and state.');
      }}
      return {{
        zipcode,
        city,
        state,
        requestedTime: normalizeDate(datetimeValue),
      }};
    }}

    function updateNavigationButtons() {{
      const now = new Date();
      const scan = currentScanTime;
      const enableNext = (hours) => {{
        if (!scan) return false;
        const target = new Date(scan.getTime() + hours * 60 * 60 * 1000);
        return target.getTime() <= now.getTime();
      }};
      navButtons.prev1.disabled = !currentScanTime;
      navButtons.prev6.disabled = !currentScanTime;
      navButtons.prev12.disabled = !currentScanTime;
      navButtons.next1.hidden = !enableNext(1);
      navButtons.next6.hidden = !enableNext(6);
      navButtons.next12.hidden = !enableNext(12);
    }}

    async function loadScan(search, scanTime = null) {{
      currentSearch = search;
      setStatus('Loading radar scan.');
      frame.removeAttribute('src');

      const response = await fetch(buildStormApiUrl(search, scanTime));
      if (!response.ok) {{
        const body = await response.text();
        throw new Error(body || `Radar request failed with status ${{response.status}}.`);
      }}
      const data = await response.json();
      currentScanTime = normalizeDate(data.timestamp);
      currentLocation = data.location;

      const iframeUrl = buildAudiomUrl(search, currentScanTime, data.location);
      frame.src = iframeUrl;
      searchPanel.hidden = true;
      mapPanel.hidden = false;
      scanSummary.textContent = `${{data.location.label}}. Radar site ${{data.site.site_id}}. Scan time ${{data.timestamp}}. ${{data.audiom_geojson.features.length}} map features.`;
      setStatus('Radar map loaded.');
      updateNavigationButtons();
      searchAgain.focus();
    }}

    async function moveHours(hours) {{
      if (!currentSearch || !currentScanTime) return;
      const nextTime = new Date(currentScanTime.getTime() + hours * 60 * 60 * 1000);
      await loadScan(currentSearch, nextTime);
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      try {{
        const search = readSearchFromForm();
        await loadScan(search, search.requestedTime);
      }} catch (error) {{
        setStatus(error.message || String(error));
      }}
    }});

    searchAgain.addEventListener('click', () => {{
      frame.removeAttribute('src');
      mapPanel.hidden = true;
      searchPanel.hidden = false;
      setStatus('');
      document.getElementById('zipcode').focus();
    }});
    navButtons.prev1.addEventListener('click', () => moveHours(-1).catch((error) => setStatus(error.message || String(error))));
    navButtons.prev6.addEventListener('click', () => moveHours(-6).catch((error) => setStatus(error.message || String(error))));
    navButtons.prev12.addEventListener('click', () => moveHours(-12).catch((error) => setStatus(error.message || String(error))));
    navButtons.next1.addEventListener('click', () => moveHours(1).catch((error) => setStatus(error.message || String(error))));
    navButtons.next6.addEventListener('click', () => moveHours(6).catch((error) => setStatus(error.message || String(error))));
    navButtons.next12.addEventListener('click', () => moveHours(12).catch((error) => setStatus(error.message || String(error))));
  </script>
</body>
</html>
"""
