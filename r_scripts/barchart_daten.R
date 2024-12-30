# Installiere und lade die benötigten Pakete
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

# Suche nach Term "Daten" und Komposita, die "Daten" enthalten.
query <- '"(?!.*([Ss]oldat|[Kk]andidat|[Uu]pdat)).*[Dd]aten.*"'

factions <- c("CDU/CSU", "SPD", "FDP", "GRUENE", "DIE LINKE", "AfD", "BSW")

# Ergebnisliste initialisieren
results <- list()

years <- as.numeric(s_attributes(gparl, "protocol_year")) # Verfügbare Jahre
years <- years[years >= 2010] # Jahre ab 2010

# Iteration über Jahre und Fraktionen
for (year in years) {
  partition_year <- partition(gparl, protocol_year = year) # Partition für ein Jahr
  for (faction in factions) {
    partition_faction <- partition(partition_year, speaker_parlgroup = faction) # Partition für Fraktion
    # Keyword-in-Context verwenden, da count() nicht wie in der PolMine-Dokumentation funktioniert; kwic leistet das Gleiche, wenn man die Zeilen zählt.
    kwic_results <- kwic(partition_faction, query = query, cqp = TRUE)
    # Wenn keine Treffer gefunden werden
    if (is.null(kwic_results)) {
      freq <- 0
    } else {
      freq <- nrow(kwic_results)
    }
    # Ergebnisse speichern
    results <- append(results, list(data.frame(
      year = year,
      fraktion = faction,
      freq = freq
    )))
  }
}

# Ergebnisse in einem Dataframe speichern
df <- do.call(rbind, results)

# Daten aggregieren und formatieren
df <- df %>%
  group_by(year, fraktion) %>%
  summarise(freq = sum(freq), .groups = "drop") # Gruppierung aufheben

# Dataframe zu einem JSON konvertieren
json_data <- toJSON(df, pretty = TRUE)

# Ergebnisse als JSON-Datei serialisieren
write(json_data, file = "factions.json")
