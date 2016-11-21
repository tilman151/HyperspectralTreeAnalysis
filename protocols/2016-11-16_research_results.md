# Protocol: 2016-11-16 (Präsentation Literaturrecherche)

## Daten
- Format: x, y, Spektralwerte (Grauwertbild)
- Ordner spacialspectral:
    - LDA -> 3 Kanäle
    - in Bereichen statistische Kenngrößen ermittelt
    - ergibt Merkmalsvektor

## Was wurde bisher erreicht
- 10-fold Cross-Validation > 90% accuracy (VNIR)
- nur spektral: 70% (VNIR)
- nur spektral: 78% (SWIR)

## Bevorzugte Methoden
- Interessante Verfahren:
    - Rotation Forest
    - Deep Learning
    - S3VM
- Standardansätze zum Vergleich: 
    - SVM
    - Random Forest

## Sonstige Notizen
- ITC-Segmentierung könnte hilfreich sein
    - z.B. Region Growing ausgehend von Baumspitzen
- erstmal keine Vitalitätslabel verfügbar
- Kombination der Features beider Kameras ist schwierig
- Presidential Vote als Ergänzung zu Ensembles möglich
- Referenzdatensätze aktuell eher unrelevant

## ToDos
- [ ] Daten herunterladen (alle)
- [ ] Framework lauffähig machen (Tilman)
- [ ] Erste Ansätze implementieren (MaTuCoVi)
- [ ] Projektplan erstellen für Christoph (alle)

