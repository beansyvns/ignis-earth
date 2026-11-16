# IGNIS Earth

**Track:** Climate · **Our own idea**

Next-day wildfire spread forecasts for Greece from free satellite and weather data, with an honest trust check.

<img width="1380" height="984" alt="forecast" src="https://github.com/user-attachments/assets/24b0e643-5b48-4791-9134-fdda436fae39" />

**Live demo:** (https://ignis-earth.onrender.com) (free server: the first load can take about a minute while it wakes up) · **Video:** [[YouTube link]](https://www.youtube.com/watch?v=2oyLXox-8yg)

## Inspiration
Wildfire maps show where fire is burning now, and weather maps show where the wind blows. Deciding where fire will reach tomorrow, and which villages to warn, is still mostly left to people combining the two. We wanted a tool that forecasts where a fire will spread tomorrow, and that admits when it doesn't know.

## What it does
IGNIS Earth splits Greece into 5 km squares. Every evening, for each square touching an active fire, it predicts the chance that fire spreads into it tomorrow.

- **Next-day spread map:** soft hazard zones around today's fire, with a replay slider of the days before.
- **Check us:** you can reveal what actually burned the next day, laid over the forecast.
- **Villages in the path:** the villages and their populations inside the high-risk zones.
- **Trust check:** every forecast is compared with the weather of past fires. Unusual days are flagged, so a crew knows when to lean on the forecast.
- **Failure days:** we show a real miss and a real false alarm, not just our best days.
- **Guided demo:** an 11-step walkthrough that opens each feature with live numbers.

## Who it's for
Fire crews and civil protection teams deciding, each evening, where to send people and which villages to warn tomorrow.

## How we built it
- **Data (all free and public):** NASA FIRMS VIIRS active fires (2014–2025), ERA5 weather via Open-Meteo (temperature, humidity, wind, gusts, rain, days since rain, dryness/VPD), Copernicus DEM elevation and slope, and GeoNames villages.
- **Features:** distance to the fire, wind blowing toward the square, uphill or downhill of the fire, fire size and intensity nearby, and how often the square burned in past years.
- **Time-based splits:** we trained on 2014–2022, tuned on 2023, and tested once on 2024–2025, which we never looked at while building.
- **Model:** logistic regression, chosen because it is simple and easy to explain.
- **Baseline:** a "downwind rule" that flags squares the wind blows toward, roughly what a person would do with a weather map.
- **Website:** a Gradio app with an interactive map, drag-and-zoom charts and a guided demo.

## Results (2024–2025, never seen during building)
| | IGNIS (logistic) | Downwind rule |
|---|---|---|
| Spread caught by the top 10 warnings a day | **52%** | 42% |
| False alarms per day | **4.4** | 8.0 |
| PR-AUC (random = 0.013) | **0.040** | 0.021 |
| Recall | 0.50 | 0.55 |

We catch slightly less spread overall, but with about half the false alarms.

**Why "top 10 warnings a day"?** Fire spreads into only about 1% of squares, so a model that says "no spread" everywhere would be 99% accurate and useless. A crew can only act on a handful of warnings each evening, so we score each method on how much real spread lands in its 10 riskiest squares per day. On the 2023 validation year, PR-AUC was 0.167 vs 0.052. The test years had only 60 squares where fire spread, so these numbers are noisy.

## Where it fails
- **A miss (2024-09-29, fire 20240929_2):** fire spread into squares the model did not rank highly.
- **A false alarm (2024-08-12, fire 20240812_1):** the model warned about squares the fire never reached.

Both are in the website's failure-days view, with the forecast laid over what actually burned.

## Challenges we ran into
- **A data leak:** our "how often this square burns" feature first included future years. We rebuilt it from training years only.
- **Fake spread:** 32% of "spread" cases were new fires 10 km away, mostly farm burns rather than real spread. We limited the forecast to squares touching today's fire.
- **Rare events:** fire spreads into only about 1% of squares, so accuracy is meaningless. We judged the model on precision at a fixed number of warnings.
- **A weak trust check:** it flags some unusual days but misses others, and we report that.

## Accomplishments we're proud of
A full pipeline from raw satellite data to an interactive forecast, an honest test on years we never touched, and a model that beats the rule a person would use while making half the false alarms.

## What we learned
Defining the problem correctly (what counts as "spread") mattered more than the choice of model. A simple, honest baseline is the bar to beat. Showing failures builds more trust than hiding them.

## What's next
- Finer squares (1 km) and hourly weather
- Fuel and vegetation maps, and live data from fire services
- Testing on other fire-prone regions (Portugal, California, Australia)
- A stronger trust check, and working with civil protection on what forecasters need

## How to run it
Online: open the live demo link above.

On your own computer (Python 3.11):
```
pip install -r requirements.txt
python app.py
```
Then open http://localhost:7860. The app reads the `demo_*` files in this folder. To rebuild everything from raw data, run `01_data.ipynb`, then `02_model.ipynb`, then `03_demo.ipynb` (made for Google Colab).

## Files
- `01_data.ipynb`: builds the dataset
- `02_model.ipynb`: trains and evaluates the models (saved with outputs)
- `03_demo.ipynb` / `app.py`: the interactive website and guided demo
- `demo_*.parquet`, `demo_*.csv`, `demo_settings.json`: the forecasts and scores the website shows
- `requirements.txt`: Python packages for the website

## Credits
Data: [NASA FIRMS (VIIRS S-NPP)](https://firms.modaps.eosdis.nasa.gov/), [ERA5 via Open-Meteo](https://open-meteo.com/), [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM), [GeoNames](https://www.geonames.org/) (CC BY 4.0). AI tool: [Claude](https://claude.ai).

AI assistance: we used Claude (Anthropic) as an AI coding assistant. It wrote most of the code from our design decisions. We chose the problem and the data, set the time-based splits, ran every experiment and tested the site.

## Licence
MIT, see [LICENSE](LICENSE).
