'use strict';
// Values transcribed from manuscript Table 1 (not recomputed from raw metrics).
const methods = [
  {name:'Restormer',setting:'Zero-shot',overall:[56.38,.438,2.456,3.503],rain:[54.69,.437,2.277,3.795],haze:[53.27,.366,1.918,3.068],snow:[61.18,.510,3.172,3.646]},
  {name:'PromptIR',setting:'Zero-shot',overall:[56.07,.443,2.437,3.491],rain:[53.48,.439,2.250,3.770],haze:[53.88,.372,1.941,3.093],snow:[60.86,.517,3.121,3.609]},
  {name:'DA-CLIP',setting:'Zero-shot',overall:[55.59,.412,2.438,3.480],rain:[52.98,.412,2.250,3.732],haze:[53.23,.325,2.014,3.071],snow:[60.57,.499,3.050,3.637]},
  {name:'DUDA · initialization',setting:'Zero-shot',overall:[56.67,.468,2.595,3.512],rain:[55.16,.454,2.510,3.777],haze:[53.73,.414,2.101,3.108],snow:[61.11,.536,3.174,3.651]},
  {name:'WResVLM',setting:'Non-blind',overall:[59.34,.456,2.640,3.574],rain:[59.80,.477,2.563,3.843],haze:[56.09,.371,2.064,3.176],snow:[62.12,.519,3.293,3.702]},
  {name:'DUDA',setting:'Blind',overall:[58.79,.494,2.702,3.593],rain:[56.15,.474,2.586,3.804],haze:[58.06,.452,2.243,3.254],snow:[62.14,.556,3.278,3.722]}
];
const captions={overall:'Overall — mean across the three evaluation sets',rain:'Rain — RealRain-2320',haze:'Haze — RTTS',snow:'Snow — Snow100K-R'};
const datasetButtons=[...document.querySelectorAll('[data-dataset]')];
datasetButtons.forEach(button=>button.addEventListener('click',()=>{
  const key=button.dataset.dataset;
  datasetButtons.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  const best=[0,1,2,3].map(i=>Math.max(...methods.map(m=>m[key][i])));
  const rows=methods.map(method=>{
    const row=document.createElement('tr');
    if(method.name==='DUDA')row.className='ours';
    const name=document.createElement('th');name.scope='row';name.textContent=method.name;row.append(name);
    const setting=document.createElement('td');setting.textContent=method.setting;row.append(setting);
    method[key].forEach((value,index)=>{const cell=document.createElement('td');cell.textContent=value.toFixed(index===0?2:3);if(value===best[index])cell.className='best';row.append(cell);});
    return row;
  });
  document.getElementById('results-body').replaceChildren(...rows);
  document.getElementById('results-caption').textContent=captions[key];
}));
const tabs=[...document.querySelectorAll('[role=tab]')];
function selectTab(tab){tabs.forEach(item=>{const active=item===tab;item.setAttribute('aria-selected',String(active));item.tabIndex=active?0:-1;document.getElementById(item.getAttribute('aria-controls')).hidden=!active;});}
tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>selectTab(tab));
  tab.addEventListener('keydown',event=>{
    let next;
    if(event.key==='ArrowDown'||event.key==='ArrowRight')next=(index+1)%tabs.length;
    if(event.key==='ArrowUp'||event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
    if(event.key==='Home')next=0;if(event.key==='End')next=tabs.length-1;
    if(next!==undefined){event.preventDefault();selectTab(tabs[next]);tabs[next].focus();}
  });
});
const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('[data-enlarge]').forEach(link=>link.addEventListener('click',event=>{
  if(!dialog.showModal)return;
  event.preventDefault();
  const img=link.querySelector('img');
  document.getElementById('dialog-image').src=img.src;
  document.getElementById('dialog-image').alt=img.alt;
  document.getElementById('dialog-title').textContent=link.dataset.enlarge;
  dialog.showModal();document.body.classList.add('modal-open');
}));
document.getElementById('close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('close',()=>document.body.classList.remove('modal-open'));
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
const navLinks=[...document.querySelectorAll('.nav-links a')];
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)navLinks.forEach(link=>{if(link.hash==='#'+entry.target.id)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}),{rootMargin:'-15% 0px -55% 0px'});navLinks.forEach(link=>observer.observe(document.querySelector(link.hash)));}
