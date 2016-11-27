# Protocol: 2016-11-24 (Framework-Tipps)

## Daten
- Gesamtbild ist verfügbar
- Training auf Gesamtbild nicht möglich, da es zu groß ist, um im Speicher gehalten zu werden
   - Sampling
   - Ausschnitte nacheinander trainieren
- Klassifikation auf Gesamtbild soll möglich sein
- (Uwe will uns noch einen Link zu einer Matlab-Bibliothek schicken, 
  die sich um das Zerteilen großer Daten zur guten Speicherausnutzung kümmert)

--- die folgenden Punkte sind Anregungen, keine Vorgaben ---

## Ergebnisse
- Ausgabe von Reports / Plots sinnvoll (mit Versionsnummer)
- bereits frühzeitig überlegen, welche Evaluierungsmethoden passend sind

## Fehleranfälligkeit
- try-catch-Blöcke
- Exception in einem Algorithmus soll nicht Gesamtexperiment unbrauchbar machen
- Zwischenergebnisse speichern

## Experimente
- XML-Datei mit Experimentablauf
  - zu jedem Datensatz Angabe, welche Vorverarbeitungsschritte, Klassifikator, Auswertung angewendet werden soll
