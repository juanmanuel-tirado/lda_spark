package ldaExercises

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer}
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Use this object template to perform an LDA analysis using Spark.
  */
object AnalysisLDA {
  def main(args:Array[String]): Unit = {

    // Time to reduce the size of the log
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    process()
  }

  def process(): Unit = {
    // Get a session
    val session = SparkSession.builder.master("local").getOrCreate()

    // Load the initial dataset
    val inputPath = "/here/goes/the/input/path"
    val dataset = session.read.textFile(inputPath)

    // Prepare dataset to be processed using LDA Helper
    // tokenizer
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W")
      .setGaps(true)
    // cleaner
    // lemmatizer -> LDA helper
    // count vectorizer
    // ...

    // get the vector counts
    val vectorizer = new CountVectorizer()
      .setInputCol("inputColumn")
      .setOutputCol("features")
      .setVocabSize(2000)

    val vectorModel = vectorizer.fit(dataset)

    // Get our vocabulary
    val vocabulary = vectorModel.vocabulary
    println(s"This is the vocabulary we are going to use: \n${vocabulary.mkString("\n")}")

    // Run LDA
    val lda = new LDA().setK(5)
    val ldaModel = lda.fit(dataset)

    // Take a look at the topics
    ldaModel.describeTopics().show()

    // Get the terms for every topic and print them
    ldaModel.describeTopics().foreach { (r: Row) => {
      val translation = r.getAs[Seq[Int]](1).map { (x: Int) => vocabulary(x) }
      println(translation.mkString(","))
      }
    }

    // Get the mixture of topics
    ldaModel.transform(dataset).show()


  }
}
