"""UI text: sources are English; Italian kicks in when the user picks it.

The app language is chosen in the Export tab and persisted in QSettings;
it applies at the next launch (widgets are built once). ``tr`` returns the
Italian from ``_IT`` when the language is ``it``, otherwise the English
source unchanged — missing entries simply stay English. ``SHORTS_STUDIO_LANG``
forces the language for tests and scripts.
"""

from __future__ import annotations

import os

_lang = os.environ.get("SHORTS_STUDIO_LANG", "en")


def lang() -> str:
    return _lang


def set_lang(code: str) -> None:
    global _lang
    _lang = code or "en"


def tr(s: str) -> str:
    return _IT.get(s, s) if _lang == "it" else s


_IT = {
    # ---- queue / preview
    "Add…": "Aggiungi…",
    "Folder…": "Cartella…",
    "Remove": "Rimuovi",
    "Find the best moments": "Trova i momenti migliori",
    "Suggests the strongest 30–60 s clips of the selected video "
    "and adds them to the list as separate shorts":
        "Propone le clip da 30–60 s più forti del video selezionato "
        "e le aggiunge alla lista come short separati",
    "How many clips to suggest": "Quante clip proporre",
    "Open project…": "Apri progetto…",
    "Save project…": "Salva progetto…",
    "Title, description & hashtags…": "Titolo, descrizione e hashtag…",
    "Thumbnails…": "Copertine…",
    "Click-worthy thumbnails for YouTube (16:9) and Shorts (9:16) from the best frames":
        "Copertine cliccabili per YouTube (16:9) e Shorts (9:16) dai fotogrammi migliori",
    "Drop videos or folders here, or use “Add”.":
        "Trascina qui video o cartelle, oppure usa “Aggiungi”.",
    "Select a video to preview": "Seleziona un video per l'anteprima",
    "Vertical preview\n\nPress “Analyze faces”": "Anteprima verticale\n\nPremi “Analizza volti”",
    "Preview audio on/off": "Audio dell'anteprima on/off",
    "Held: you hear the original audio. Released: the audio as it will be exported "
    "(cinematic audio and music, if on)":
        "Premuto: ascolti l'audio originale. Rilasciato: l'audio come uscirà "
        "(audio cinematico e musica, se attivi)",
    "Skip cuts": "Salta i tagli",
    "While playing, skip the parts that will be cut (in red)":
        "In riproduzione salta le parti che verranno tagliate (in rosso)",
    "Analyze faces": "Analizza volti",
    "Transcribe": "Trascrivi",
    "Whisper: needed to see fillers, retakes and captions in the preview "
    "(the result is reused by the render)":
        "Whisper: serve per vedere filler, ripetizioni e sottotitoli "
        "nell'anteprima (il risultato viene riusato dal render)",
    "Captions…": "Sottotitoli…",
    "Fix the caption text before burning it into the video":
        "Correggi il testo dei sottotitoli prima di imprimerli nel video",
    "Enhancement": "Miglioramento",
    "Show the image enhancement in the vertical preview":
        "Mostra il miglioramento immagine nell'anteprima verticale",
    "Auto from here": "Auto da qui",
    "From this frame on, the crop follows the face by itself again":
        "Da questo frame il crop torna a seguire il volto da solo",
    "Clear corrections": "Annulla correzioni",
    "Safe zones: off": "Zone sicure: no",
    "Show the areas covered by the platform's interface":
        "Mostra le aree coperte dall'interfaccia della piattaforma",
    "Undo (⌘Z): crop corrections, people, captions, title":
        "Annulla (⌘Z): correzioni del crop, persone, sottotitoli, titolo",
    "Redo (⇧⌘Z)": "Ripeti (⇧⌘Z)",
    "Who to follow:": "Chi seguire:",
    "Save…": "Salva…",
    "Delete": "Elimina",
    "Opening title": "Titolo iniziale",
    "Edit captions": "Modifica sottotitoli",
    "Person choice": "Scelta persona",
    "Crop correction": "Correzione inquadratura",
    "Back to auto": "Torna automatico",
    "Clear all corrections": "Cancella correzioni",
    "Undo: ": "Annulla: ",
    "Nothing to undo": "Niente da annullare",
    "Redo: ": "Ripeti: ",
    "Nothing to redo": "Niente da ripetere",
    "Undone": "Annullato",
    "Redone": "Ripristinato",
    "Person %d · in %d shots · ~%ds on screen":
        "Persona %d · in %d inquadrature · ~%ds a schermo",
    "Use the “Follow” setting (main person or whoever speaks)":
        "Usa l'impostazione “Segui” (persona principale o chi parla)",

    # ---- crop tab
    "Follow the face vertically too": "Segui il volto anche in verticale",
    "Auto punch-in on phrases": "Punch-in automatico sulle frasi",
    "Aspect": "Formato",
    "Follow": "Segui",
    "Layout": "Layout",
    "Smoothing": "Fluidità",
    "Dead zone": "Zona morta",
    "Max pan speed": "Velocità pan max",
    "Face height": "Altezza volto",
    "Punch-in zoom": "Zoom punch-in",
    "Wide shots": "Piani larghi",
    "Wide when face <": "Largo se volto <",
    "Max wide zoom": "Zoom max larghi",
    "Detection": "Rilevamento",
    "Main person": "Persona principale",
    "Whoever speaks": "Chi parla",
    "One person": "Una persona",
    "Split 2 people (top/bottom)": "Split 2 persone (sopra/sotto)",
    "Zoom on subject": "Zoom sul soggetto",
    "Full height": "Altezza piena",
    "Every frame (slow)": "Ogni frame (lento)",
    "Every 2 frames": "Ogni 2 frame",
    "Every 3 frames (fast)": "Ogni 3 frame (veloce)",
    "Drag the yellow box in the preview to correct the crop from that point on. "
    "Below the preview, pick who to follow (select every thumbnail of the same "
    "person). “Whoever speaks” cuts to who is talking.":
        "Trascina il riquadro giallo nell'anteprima per correggere il crop da quel "
        "punto. Sotto l'anteprima scegli chi seguire (seleziona tutte le miniature "
        "della stessa persona). “Chi parla” stacca su chi sta parlando.",

    # ---- cuts tab
    "Cut pauses": "Taglia le pause",
    "Cut breaths": "Taglia i respiri",
    "Cut ums / uhs  (Whisper)": "Taglia ehm / uhm  (Whisper)",
    "Keep only the last retake  (Whisper)": "Tieni solo l'ultima ripetizione  (Whisper)",
    "Min pause": "Pausa minima",
    "Padding": "Margine attorno",
    "Voice threshold": "Soglia voce",
    "Cut parts are red on the timeline. Fillers and retakes appear after “Transcribe”. "
    "Whisper runs once per video and the result stays cached.":
        "Le parti tagliate sono in rosso sulla timeline. Filler e ripetizioni "
        "compaiono dopo “Trascrivi”. Whisper gira una volta per video e il "
        "risultato resta in cache.",
    "Cuts": "Tagli",

    # ---- audio tab
    "Cinematic audio": "Audio cinematico",
    "No music": "Nessuna musica",
    "Noise reduction": "Riduzione rumore",
    "De-reverb": "Riduzione riverbero",
    "Warmth": "Calore",
    "Presence": "Presenza",
    "Air": "Aria",
    "Denoise, de-esser, compression, EQ and loudness at −14 LUFS.":
        "Denoise, de-esser, compressione, EQ e loudness a −14 LUFS.",
    "Background music": "Musica di sottofondo",
    "Music volume": "Volume musica",
    "Duck under voice": "Abbassa sotto la voce",
    "The music loops for the whole duration and ducks by itself when someone talks.":
        "La musica va in loop per tutta la durata e si abbassa da sola quando "
        "qualcuno parla.",
    "Light": "Leggero",
    "Medium": "Medio",
    "Strong": "Forte",

    # ---- quality tab
    "Face restoration": "Restauro volti",
    "Model": "Modello",
    "Strength": "Intensità",
    "Fidelity (CodeFormer)": "Fedeltà (CodeFormer)",
    "Rebuilds the face (eyes, skin, teeth) and blends it back into the frame with "
    "feathered edges. Time per face per frame on an M1: 1 minute of short "
    "≈ 4 min with GPEN 256, ≈ 2 hours with CodeFormer. High fidelity = closer "
    "to the original. The first run downloads the model (75–380 MB). CodeFormer and "
    "GPEN are free for non-commercial use only; GFPGAN is fine for work. "
    "Preview it with “Enhancement” on, on a paused frame.":
        "Ricostruisce il volto (occhi, pelle, denti) e lo rimette nel fotogramma con "
        "i bordi sfumati. Tempi per volto e per fotogramma su un M1: 1 minuto di short "
        "≈ 4 min con GPEN 256, ≈ 2 ore con CodeFormer. Fedeltà alta = più simile "
        "all'originale. Il primo uso scarica il modello (75–380 MB). CodeFormer e "
        "GPEN sono gratuiti solo per uso non commerciale; GFPGAN anche per lavoro. "
        "Nell'anteprima si vede con “Miglioramento” attivo, a video fermo.",
    "Normalize skin color": "Normalizza il colore della pelle",
    "Save as .cube LUT…": "Salva come LUT .cube…",
    "The same correction as a 3D LUT, for DaVinci Resolve":
        "La stessa correzione come LUT 3D, da usare in DaVinci Resolve",
    "Face beauty": "Beauty sul volto",
    "Beauty": "Beauty",
    "Skin smoothing": "Levigatura pelle",
    "Spots and blemishes": "Macchie e brufoli",
    "Wrinkles and eye bags": "Rughe e occhiaie",
    "Only on face skin: eyes, eyebrows, lips, beard and hair stay sharp. "
    "Smoothing that keeps some grain (no plastic look), spots replaced by the "
    "skin around them, wrinkles softened and eye bags lightened. ~10–25 ms per face.":
        "Solo sulla pelle del viso: occhi, sopracciglia, labbra, barba e capelli restano "
        "nitidi. Levigatura che conserva un po' di grana (niente effetto plastica), "
        "macchie sostituite dalla pelle attorno, rughe attenuate e occhiaie schiarite. "
        "Costo ~10–25 ms per volto.",
    "Skin": "Pelle",
    "Measures cheeks and forehead, then pulls the tint back to the natural skin line "
    "and removes excess saturation, only on skin colours: red clothes and background "
    "stay as they were. Useful for cameras that make skin too red.":
        "Misura guance e fronte, poi riporta la tinta sulla linea naturale della pelle "
        "e toglie la saturazione in eccesso, solo sui colori della pelle: vestiti "
        "rossi e sfondo restano com'erano. Utile per camere che fanno la pelle troppo rossa.",
    "Auto white balance & exposure": "Bilanciamento bianco ed esposizione automatici",
    "Stabilize shaky footage": "Stabilizza le riprese mosse",
    "Colour": "Colore",
    "Look": "Look",
    "Look strength": "Intensità look",
    "Auto balance evens out colour and exposure: handy to give shorts of the same "
    "series the same look.":
        "Il bilanciamento automatico pareggia colore ed esposizione: utile per avere "
        "short della stessa serie con lo stesso aspetto.",
    "Stabilization": "Stabilizzazione",
    "Measures camera shake and compensates by shifting the crop (plus a slight "
    "rotation). The first run analyses the whole video.":
        "Misura il tremolio della camera e lo compensa spostando il ritaglio (più una "
        "leggera rotazione). La prima volta analizza tutto il video.",
    "Sharpening & upscale": "Nitidezza e upscale",
    "Sharpening": "Nitidezza",
    "Extra on face": "Extra sul volto",
    "Works only on the vertical crop. “Fast” costs almost nothing.":
        "Lavora solo sul ritaglio verticale. “Veloce” costa quasi zero.",
    "Off": "Spento",
    "Fast (sharpen + face)": "Veloce (nitidezza + volto)",
    "GPEN 256 — fast (~0.15 s/fr.)": "GPEN 256 — veloce (~0,15 s/fot.)",
    "GPEN 512 — detailed (~1.8 s/fr.)": "GPEN 512 — dettagliato (~1,8 s/fot.)",
    "GFPGAN 1.4 — commercial use ok (~2 s/fr.)": "GFPGAN 1.4 — uso commerciale ok (~2 s/fot.)",
    "CodeFormer — top quality (~5 s/fr.)": "CodeFormer — qualità massima (~5 s/fot.)",
    "Quality": "Qualità",

    # ---- effects tab
    "Handheld camera": "Camera a mano",
    "Slow push on every phrase": "Spinta lenta su ogni frase",
    "Crash zoom on strong words": "Crash zoom sulle parole forti",
    "Shake on voice peaks": "Scossone sui picchi di voce",
    "Camera moves": "Movimenti di camera",
    "Cut transitions": "Transizioni sui tagli",
    "Effect": "Effetto",
    "When": "Quando",
    "Min distance": "Distanza minima",
    "Every cut": "Tutti i tagli",
    "Every other one": "Uno sì e uno no",
    "Only shot changes": "Solo cambi d'inquadratura",
    "Crash zoom uses the transcript's strong words (questions, exclamations, "
    "numbers, “never”, “secret”…) or, without a transcript, voice peaks. Cuts "
    "are the shot changes and the pause cuts. Everything is previewed in play; "
    "export cost is nearly zero.":
        "Il crash zoom usa le parole forti della trascrizione (domande, esclamazioni, "
        "numeri, “mai”, “segreto”…) o, senza trascrizione, i picchi di voce. I tagli "
        "sono gli stacchi d'inquadratura e i tagli delle pause. Tutto si vede "
        "nell'anteprima in play; costo in export quasi nullo.",
    "None": "Nessuna",
    "Random": "Casuale",
    "Effects": "Effetti",

    # ---- captions tab
    "Word by word": "Parola per parola",
    "Whole line": "Riga intera",
    "UPPERCASE": "Maiuscolo",
    "Captions": "Sottotitoli",
    "Style": "Stile",
    "Language": "Lingua",
    "Words per line": "Parole per riga",
    "Max line duration": "Durata max riga",
    "Vertical position": "Posizione verticale",
    "Text colour": "Colore testo",
    "Active word colour": "Colore parola attiva",
    "Original language": "Lingua originale",
    "English (translation)": "Inglese (traduzione)",
    "Original + English below": "Originale + inglese sotto",
    "Caption language": "Lingua sottotitoli",
    "Whisper does the English translation, offline, from any language.":
        "La traduzione in inglese la fa Whisper, offline, da qualsiasi lingua.",
    "Automatic": "Automatica",
    "Auto (first sentence)": "Automatico (prima frase)",
    "Written by me (per video)": "Scritto da me (per video)",
    "Opening title (hook)": "Titolo iniziale (hook)",
    "Title": "Titolo",
    "Text": "Testo",
    "Duration": "Durata",
    "Opening title for the selected video": "Titolo iniziale per il video selezionato",
    "e.g. Follow me for more videos!  (empty = none)":
        "es. Seguimi per altri video!  (vuoto = niente)",
    "Outro": "Chiusura",
    "Closing CTA": "Invito finale",
    "Logo": "Logo",
    "Image (PNG)": "Immagine (PNG)",
    "No logo": "Nessun logo",
    "Top right": "In alto a destra",
    "Top left": "In alto a sinistra",
    "Bottom right": "In basso a destra",
    "Bottom left": "In basso a sinistra",
    "Position": "Posizione",
    "Size": "Dimensione",
    "Opacity": "Opacità",
    "White": "Bianco",
    "Yellow": "Giallo",
    "Green": "Verde",
    "Cyan": "Ciano",
    "Pink": "Rosa",
    ".srt file only": "Solo file .srt",
    "Burned in video + .srt": "Impressi nel video + .srt",
    "Texts": "Testi",

    # ---- export tab
    "4:5 (Instagram feed)": "4:5 (feed Instagram)",
    "1:1 (square)": "1:1 (quadrato)",
    "16:9 (YouTube landscape)": "16:9 (YouTube orizzontale)",
    "Next to the source video": "Accanto al video originale",
    "Clear cache": "Svuota cache",
    "Resolution": "Risoluzione",
    "Encoding": "Encoding",
    "Quality (CRF)": "Qualità (CRF)",
    "Also export as:": "Esporta anche in:",
    "Output folder": "Cartella output",
    "File suffix": "Suffisso file",
    "Auto (hardware on Mac)": "Automatico (hardware sul Mac)",
    "Also create the thumbnail (JPG)": "Crea anche la copertina (JPG)",
    "Create title, description and hashtags (.txt)": "Crea titolo, descrizione e hashtag (.txt)",
    "Encoder": "Encoder",
    "Analysis cache": "Cache analisi",
    "Extra formats reuse the same analysis: they only cost the render. The Mac "
    "hardware encoder is 3–5× faster. Lower CRF = higher quality and bigger files "
    "(with hardware it sets the bitrate).":
        "I formati extra riusano la stessa analisi: costano solo il render. "
        "L'encoder hardware del Mac è 3–5 volte più veloce. CRF più basso = qualità "
        "più alta e file più grande (con l'hardware regola il bitrate).",
    "UI language": "Lingua interfaccia",
    "Restart the app to apply the new language.":
        "Riavvia l'app per applicare la nuova lingua.",
    "Slow PC": "PC lento",
    "720 × 1280 (slow PC)": "720 × 1280 (PC lento)",
    "Fast": "Veloce",
    "Balanced": "Bilanciato",

    # ---- bottom bar / batch
    "Ready": "Pronto",
    "Create short": "Crea short",
    "Create %d shorts": "Crea %d short",
    "Done": "Fatto",
    "Stopping…": "Interruzione…",
    "Video %d/%d · %s %d%%": "Video %d/%d · %s %d%%",
    "✓ Created: %s": "✓ Creato: %s",
    "✕ Error: %s": "✕ Errore: %s",
    "Error: %s": "Errore: %s",
    "cancelled": "annullato",
    "Processing. Stop and quit?": "Elaborazione in corso. Interrompere e uscire?",
    "Resume the previous session (%d videos)?": "Riprendere la sessione precedente (%d video)?",
    "%d shots (%d wide) · face on %d%% of frames":
        "%d inquadrature (%d larghe) · volto su %d%% dei frame",
    " · split on %d%% of frames": " · split su %d%% dei frame",
    " · split: need 2 people in the same shot": " · split: servono 2 persone nella stessa inquadratura",
    " · stabilized": " · stabilizzato",
    " · cuts: −%.1fs": " · tagli: −%.1fs",
    " · translation: done at the first export (or with “Transcribe”)":
        " · traduzione: si fa al primo export (o con “Trascrivi”)",
    "Press “Analyze faces” to see the vertical crop.":
        "Premi “Analizza volti” per vedere il crop verticale.",

    # ---- dialogs / logs
    "Save preset": "Salva preset",
    "Preset name:": "Nome del preset:",
    "Only presets you saved yourself (★) can be deleted.":
        "Si possono eliminare solo i preset salvati da te (★).",
    "Preset “%s” applied": "Preset “%s” applicato",
    "Background music": "Musica di sottofondo",
    "Audio (%s)": "Audio (%s)",
    "Output folder": "Cartella output",
    "Cache cleared": "Cache svuotata",
    "Add videos": "Aggiungi video",
    "Video (%s)": "Video (%s)",
    "Add every video in a folder": "Aggiungi tutti i video di una cartella",
    "Heavy video: making a light copy for smooth preview…":
        "Video pesante: preparo una copia leggera per l'anteprima fluida…",
    "Proxy discarded (frame count differs from the original)":
        "Proxy scartato (fotogrammi diversi dall'originale)",
    "Preview: playing the light copy": "Anteprima: uso la copia leggera in riproduzione",
    "Preview audio unavailable (%s)": "Audio dell'anteprima non disponibile (%s)",
    "Preview audio: %s": "Audio dell'anteprima: %s",
    "Audio: as exported": "Audio: come uscirà",
    "You're hearing the original": "Stai ascoltando l'originale",
    "You're hearing the treated audio": "Stai ascoltando l'audio trattato",
    "Analysing %s: %d frames": "Analisi di %s: %d frame",
    "Face analysis": "Analisi volti",
    "Transcript already cached": "Trascrizione già in cache",
    "Transcript: %d words": "Trascrizione: %d parole",
    "First I'll transcribe the video, then I'll open the caption editor…":
        "Prima trascrivo il video, poi apro l'editor dei sottotitoli…",
    "Select a video in the list first.": "Seleziona prima un video nella lista.",
    "Select the original video, not a clip.": "Seleziona il video originale, non una clip.",
    "No transcript: can't look for the best moments":
        "Nessuna trascrizione: impossibile cercare i momenti migliori",
    "Video too short or no speech: no moments suggested":
        "Video troppo corto o senza parlato: nessun momento proposto",
    "Found %d moments in %s: they're in the list under the video":
        "Trovati %d momenti in %s: li trovi in lista sotto al video",
    "Stabilization: motion analysis (once per file)…":
        "Stabilizzazione: analisi del movimento (una volta per file)…",
    "Missing: %s": "Manca: %s",
    "Save project": "Salva progetto",
    "Shorts Project (*.shorts)": "Progetto Shorts (*.shorts)",
    "Saving failed: %s": "Salvataggio non riuscito: %s",
    "Project saved: %s": "Progetto salvato: %s",
    "Open project": "Apri progetto",
    "Can't open the project: %s": "Impossibile aprire il progetto: %s",
    "Project opened: %s": "Progetto aperto: %s",
    "invalid project file": "file di progetto non valido",
    "Select a video.": "Seleziona un video.",
    "You need a transcript first: press “Transcribe”.":
        "Prima serve la trascrizione: premi “Trascrivi”.",
    "Social texts": "Testi per i social",
    "Copy": "Copia",
    "Close": "Chiudi",
    "Press “Analyze faces” first: thumbnails start from the frames with the best face.":
        "Prima premi “Analizza volti”: le copertine partono dai fotogrammi con il volto migliore.",
    "Thumbnails: picking frames and cutting out the subject…":
        "Copertine: scelgo i fotogrammi e scontorno…",
    "Thumbnails": "Copertine",
    "No usable face for a thumbnail.": "Nessun volto utilizzabile per una copertina.",
    "Add at least one video.": "Aggiungi almeno un video.",
    "Qt captions unavailable (%s), using OpenCV": "Sottotitoli Qt non disponibili (%s), uso OpenCV",
    "Correction at %s": "Correzione a %s",
    "Enable “Normalize skin color” and press “Analyze faces”: the LUT is based on "
    "the skin measured in the video.":
        "Attiva “Normalizza il colore della pelle” e premi "
        "“Analizza volti”: la LUT si basa sulla pelle misurata nel video.",
    "Save LUT": "Salva LUT",
    "LUT saved: %s (in Resolve: right-click the clip > LUT)":
        "LUT salvata: %s (in Resolve: tasto destro sulla clip > LUT)",
    "Measuring skin…": "Misuro la pelle…",
    "No measurable face in this video.": "Nessun volto misurabile in questo video.",
    "Skin measure": "Misura pelle",
    "Face restored": "Volto restaurato",
    "Restoring faces…": "Restauro volti…",
    "Enhanced": "Migliorata",
    "Preview": "Anteprima",
    "With %s face restoration will take about %s (M1 estimate, one face per "
    "frame; cuts reduce it).\n\nContinue?":
        "Con %s il restauro volti richiederà circa %s (stima su un M1, un volto per "
        "fotogramma; i tagli lo riducono).\n\nContinuare?",
    "%s: following %s": "%s: segue %s",
    "Auto": "Auto",
    "%d lines edited.": "%d righe modificate.",
    "Discard all corrections and go back to the Whisper transcript?":
        "Annullare tutte le correzioni e tornare alla trascrizione di Whisper?",
    "Double-click a row to fix it (Enter to confirm, empty text = deletes "
    "the row). Click a row to see it in the preview.":
        "Doppio clic su una riga per correggerla (Invio per confermare, testo vuoto = elimina "
        "la riga). Clic su una riga per vederla nell'anteprima.",
    "person %d": "persona %d",
    "Saved: %s": "Salvata: %s",
    "Saved %d thumbnails to %s": "Salvate %d copertine in %s",
    "2-4 strong words": "2-4 parole forti",
    "Highlighted word": "Parola colorata",
    "none": "nessuna",
    "Pick a thumbnail and press “Save”": "Scegli una copertina e premi “Salva”",
    "Save all": "Salva tutte",
    "Save": "Salva",
    "Captions — ": "Sottotitoli — ",
    "Find (e.g. a misspelled name)": "Trova (es. un nome sbagliato)",
    "Replace with": "Sostituisci con",
    "Whole word": "Parola intera",
    "Replace all": "Sostituisci tutto",
    "Start": "Inizio",
    "End": "Fine",
    "%d lines · %d words": "%d righe · %d parole",
    "Restore original transcript": "Ripristina trascrizione originale",
    "skin measured: hue %.0f°, saturation %.0f — already natural, no correction":
        "pelle misurata: tinta %.0f°, saturazione %.0f — già naturale, nessuna correzione",
    "skin measured: hue %.0f° → %.0f° (natural %.0f–%.0f°), saturation %.0f → %.0f":
        "pelle misurata: tinta %.0f° → %.0f° (naturale %.0f–%.0f°), saturazione %.0f → %.0f",

    # ---- jobs (export pipeline logs)
    "  %d words (%s)": "  %d parole (%s)",
    "  no speech detected: pauses not cut": "  nessun parlato rilevato: pause non tagliate",
    "  %d breaths": "  %d respiri",
    "  %d fillers (um, uh…)": "  %d filler (ehm, uhm…)",
    "  %d retakes": "  %d ripetizioni",
    "Trim": "Ritaglio",
    "  video has no audio: skipping cuts, captions, punch-in and cinematic audio":
        "  il video non ha audio: salto tagli, sottotitoli, punch-in e audio cinematico",
    "Transcribing (Whisper)…": "Trascrizione (Whisper)...",
    "  transcript from cache": "  trascrizione dalla cache",
    "Transcription": "Trascrizione",
    "  Whisper unavailable (%s): skipping fillers, retakes and captions":
        "  Whisper non disponibile (%s): salto filler, ripetizioni e sottotitoli",
    "  nothing to cut": "  niente da tagliare",
    "Cuts: %d segments, %.1fs of %.1fs": "Tagli: %d segmenti, %.1fs su %.1fs",
    "Cinematic audio…": "Audio cinematico...",
    "Faces: reusing the preview analysis": "Volti: uso l'analisi dell'anteprima",
    "Detecting faces…": "Rilevamento volti...",
    "Faces": "Volti",
    "Framing: follows the people picked in the preview":
        "Inquadratura: segue le persone scelte nell'anteprima",
    "Framing: %d manual corrections": "Inquadratura: %d correzioni manuali",
    "Skin: %s": "Pelle: %s",
    "no face to measure": "nessun volto da misurare",
    "Downloading the %s model (%d MB, once)…": "Scarico il modello %s (%d MB, una volta sola)...",
    "Model download": "Download modello",
    "Beauty: smoothing %.0f%%, blemishes %.0f%%, wrinkles %.0f%%":
        "Beauty: levigatura %.0f%%, macchie %.0f%%, rughe %.0f%%",
    "Face restoration %s: %.2f s per face -> about %s for %d frames":
        "Restauro volti %s: %.2f s per volto -> circa %s per %d fotogrammi",
    "English translation (Whisper)…": "Traduzione in inglese (Whisper)...",
    "Translation": "Traduzione",
    "  translation unavailable (%s)": "  traduzione non disponibile (%s)",
    "Captions: %d lines -> %s": "Sottotitoli: %d righe -> %s",
    "Colour: look %s%s": "Colore: look %s%s",
    ", auto balance": ", bilanciamento automatico",
    "Stabilization: motion analysis…": "Stabilizzazione: analisi del movimento...",
    "  face on %d/%d frames, %d shots": "  volto su %d/%d frame, %d inquadrature",
    "  split screen on %d frames": "  split screen su %d frame",
    "  moves: %d cuts": "  movimenti: %d stacchi",
    "Music": "Musica",
    "Thumbnail: %s": "Copertina: %s",
    "  thumbnail not created (%s)": "  copertina non creata (%s)",
    "YouTube thumbnail: %s": "Copertina YouTube: %s",
    "  YouTube thumbnail not created (%s)": "  copertina YouTube non creata (%s)",
    "Social texts: %s": "Testi social: %s",
    "%.1f hours": "%.1f ore",
    "audio preview: %s": "anteprima audio: %s",
    "unreadable frame": "fotogramma non leggibile",
    "Thumbnails: downloading the cutout model (4.6 MB, once)…":
        "Copertine: scarico il modello di scontorno (4,6 MB, una volta sola)…",
    "Thumbnails: cutout unavailable (%s), using a soft shape":
        "Copertine: scontorno non disponibile (%s), uso una sagoma morbida",
    "Pop (blurred background)": "Pop (sfondo sfocato)",
    "Bright colour": "Colore acceso",
    "Close-up": "Primo piano",
    "Dark with glow": "Scuro con alone",
    "Interview": "Intervista",
    "2-person podcast": "Podcast 2 persone",
    "Split-screen podcast": "Podcast split screen",
    "Natural": "Naturale",
    "Warm": "Caldo",
    "Cool": "Freddo",
    "Cinematic (teal & orange)": "Cinematico (teal & orange)",
    "Pop (contrast & colour)": "Pop (contrasto e colore)",
    "B&W": "Bianco e nero",
    "After “Transcribe”, the captions show in the vertical preview.":
        "Dopo “Trascrivi” i sottotitoli si vedono nell'anteprima verticale.",
    "Preview proxy": "Proxy anteprima",
    "Preview audio": "Audio anteprima",
    "White balance": "Bilanciamento",
    "Images (*.png *.jpg *.jpeg *.webp)": "Immagini (*.png *.jpg *.jpeg *.webp)",
    "unreadable logo: %s": "logo non leggibile: %s",
    "proxy not created: %s": "proxy non creato: %s",
    "model %s not downloaded": "modello %s non scaricato",
    "— choose —": "— scegli —",
    "Wide · zoom %.2f×": "Largo · zoom %.2f×",
    "No face": "Nessun volto",
    "✂ cut · ": "✂ tagliato · ",
}
