const byId = (id) => document.getElementById(id);
const input = byId('image');
let imageFile = null;
let previewUrl = null;
let busy = false;
input.addEventListener('change', () => {
  imageFile = input.files[0] || null;
  if (imageFile && imageFile.size > 10 * 1024 * 1024) {
    imageFile = null;
    input.value = '';
    byId('status').textContent = 'Choose an image smaller than 10 MB.';
    byId('generate').disabled = true;
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  byId('preview').hidden = !imageFile;
  if (imageFile) { previewUrl = URL.createObjectURL(imageFile); byId('preview').src = previewUrl; }
  byId('generate').disabled = !imageFile || busy;
  byId('result').hidden = true;
  byId('status').textContent = imageFile ? 'Image ready. Choose a model and generate a caption.' : 'Choose an image to start.';
});
byId('model').addEventListener('change', () => {
  byId('model-note').textContent = byId('model').value === 'generic'
    ? 'This nonvisual control returns the same sentence for every image. It checks the pipeline and does not describe image content.'
    : 'A pretrained image captioner from Salesforce. The first request downloads the model weights.';
});
byId('generate').addEventListener('click', async () => {
  if (!imageFile || busy) return;
  busy = true;
  input.disabled = true;
  byId('model').disabled = true;
  byId('generate').disabled = true;
  byId('status').className = '';
  byId('status').textContent = 'Generating caption. The first model download can take several minutes…';
  byId('result').hidden = true;
  try {
    const response = await fetch(`/api/caption?model=${encodeURIComponent(byId('model').value)}`, {method:'POST',body:imageFile});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The request failed. Try again.');
    byId('caption-text').textContent = data.caption;
    const latency = data.latency_seconds;
    byId('metadata').textContent = `${data.model_id || data.model} · ${latency == null ? 'Timing unavailable' : Number(latency).toFixed(3) + ' s inference'}${data.device ? ' · ' + data.device : ''}`;
    byId('result').hidden = false;
    byId('status').textContent = data.visual_input ? 'Caption ready.' : 'Control output ready. This caption does not use the image.';
  } catch (error) { byId('status').textContent = error.message; byId('status').className = 'error'; }
  finally { busy = false; input.disabled=false; byId('model').disabled=false; byId('generate').disabled = !imageFile; }
});
function element(tag, text, className) { const node=document.createElement(tag); if(text!=null) node.textContent=text; if(className) node.className=className; return node; }
async function gallery() {
  byId('gallery-status').textContent = 'Loading saved comparisons…';
  try {
    const response=await fetch('/api/gallery'); const rows=await response.json();
    if(!response.ok) throw new Error(rows.detail || 'The gallery could not load.');
    byId('gallery-items').replaceChildren();
    byId('gallery-status').textContent=rows.length ? `${rows.length} test images. Results belong to the configured run.` : 'No run is configured. Start the server with --manifest and --predictions to view a comparison.';
    for(const row of rows) {
      const article=element('article',null,'gallery-row'); const img=element('img'); img.src=row.image_url; img.alt=`Test image ${row.image_id}`; img.loading='lazy'; article.append(img);
      const details=element('div'); details.append(element('h3',row.image_id)); details.append(element('p',row.evidence_label || 'Configured test sample','privacy'));
      for(const pred of row.predictions) { const p=element('p',null,'prediction'); p.append(element('small',pred.model==='generic' ? 'Generic nonvisual control' : pred.model)); p.append(document.createTextNode(pred.caption)); details.append(p); }
      details.append(element('h4','Human reference captions')); const list=element('ul'); for(const ref of row.references) list.append(element('li',ref)); details.append(list); article.append(details); byId('gallery-items').append(article);
    }
  } catch(error) { byId('gallery-status').textContent=error.message; }
}
for(const tab of ['demo','gallery']) byId(`${tab}-tab`).addEventListener('click',()=>{
  for(const name of ['demo','gallery']) { const active=name===tab; byId(name).hidden=!active; byId(`${name}-tab`).classList.toggle('active',active); byId(`${name}-tab`).setAttribute('aria-selected',String(active)); }
  if(tab==='gallery') gallery();
});
