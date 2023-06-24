using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace ClassLibrary_arm
{
    public class Program
    {
        public float Comp(string path1, int windowSize, int seriesLength, int trainSize, int horizon)
        {
            // создаем новый MLContext
            var mlContext = new MLContext();
            // загружаем данные в IDataView
            var dataView = mlContext.Data.LoadFromTextFile<InputData>(@path1, separatorChar: ';');
            // Разделяем данные на обучающие и тестовые наборы
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3);
           
            var pipeline = mlContext.Forecasting.ForecastBySsa(
                nameof(OutputData.ForecastedPrice),
                nameof(InputData.price),
                windowSize,
                seriesLength,
                trainSize, horizon);

            // обучаем модель и делаем прогнозы
            var model = pipeline.Fit(trainTestSplit.TrainSet);

            var forecaster = model.CreateTimeSeriesEngine<InputData, OutputData>(mlContext);
            var forecast = forecaster.Predict();

            // выводим прогнозируемые цены
            //for (int i = 0; i < forecast.ForecastedPrice.Length; i++)
            //{
            //    Console.WriteLine("Forecasted price for month {0}: {1}", i + 1, forecast.ForecastedPrice[i]);
            //}
            float price_prog = forecast.ForecastedPrice[horizon];
            return price_prog;
        }
        public class InputData
        {
            [LoadColumn(0), ColumnName("date")]
            public DateTime date;

            [LoadColumn(1), ColumnName("kol")]
            public float kol;

            [LoadColumn(2), ColumnName("price")]
            public float price;
        }

        public class OutputData
        {
            [ColumnName("ForecastedPrice")]
            public float[] ForecastedPrice;
        }

    }
}
