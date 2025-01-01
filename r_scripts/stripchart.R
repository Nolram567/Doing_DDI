#Installiere und lade die benötigten Pakete
install.packages("polmineR")
install.packages("dplyr")
install.packages("tidyr")
install.packages("jsonlite")
install.packages("devtools")

library(polmineR)
library(dplyr)
library(tidyr)
library(jsonlite)
library(devtools)

# Lade die Development-Version, da cwbtools derzeit im cran nicht verfügbar ist
devtools::install_github("PolMine/cwbtools", ref = "dev", force = TRUE) 

library(cwbtools)

# Das Korpus muss im Registry-Verzeichnis liegen und die Umgebungsvariable muss definiert sein.
gparl <- corpus("GERMAPARL2")

# Such nach dem Term "Daten" und nach Komposita, die den Term "Daten" enthalten.
query <- '"(?!.*([Ss]oldat|[Kk]andidat|[Uu]pdat)).*[Dd]aten.*"'

p <- partition(gparl, protocol_lp = c("17", "18", "19", "20"))

# Alle verfügbaren Parlamentarier:innen ermitteln, die in den spezifizierten Legislaturperioden sprechen.
speakers <- s_attributes(p, "speaker_name")

# Ergebnisse als Liste initialisieren
results <- list()

# Iteration über alle Redner:innen
for (speaker in speakers) {
  
  partition_lp <- partition(gparl, protocol_lp = c("17", "18", "19", "20"))
  
  # Partition für eine:n spezifische:n Parlamentarier:in erstellen
  partition_speaker <- partition(partition_lp, speaker_name = speaker)
  
  # KWIC-Abfrage durchführen
  kwic_results <- kwic(partition_speaker, query = query, cqp = TRUE)
  
  # Treffer zählen
  freq <- if (is.null(kwic_results)) 0 else nrow(kwic_results)
  
  # Zugehörige Fraktion/Partei der/des Sprecher:in ermitteln
  sp_parties <- s_attributes(partition_speaker, "speaker_party")
  sp_party <- if (length(sp_parties) > 0) sp_parties[1] else NA
  
  # Ergebnisse abspeichern
  results <- append(results, list(data.frame(
    speaker = speaker,
    fraktion = sp_party,
    freq = freq
  )))
}

# Ergebnisse in einen Dataframe umwandeln
df <- do.call(rbind, results)

# Personen entfernen, die weniger als 5-mal den Term "Daten" gebraucht haben
df <- df %>% filter(freq >= 5)

# Dataframe zu JSON konvertieren
json_data <- toJSON(df, pretty = TRUE)

# Ergebnisse als JSON-Datei serialisieren
write(json_data, file = "parlamentarians.json")
