package ldaExercises

import edu.stanford.nlp.process.Morphology
import edu.stanford.nlp.simple.Document
import scala.collection.JavaConversions._


/**
  * This is a helper object to prepare LDA spark input datasets.
  */
object LDAHelper {

  def filterSpecialCharacters(document: String) = document.replaceAll("""[! @ # $ % ^ & * ( ) _ + - − , " ' ; : . ` ? --]""", " ")

  def removeDoubleSpaces(document: String) = {
    val reg = """[\n\t\p{Zs}]+"""
    document.replaceAll(reg, " ")
  }

  def getStemmedText(document: String) = {
    val morphology = new Morphology()
    new Document(document).sentences().toList.flatMap(_.words().toList.map(morphology.stem)).mkString(" ")
  }

  def getLemmaText(document: String, morphology: Morphology) = {
    val string = new StringBuilder()
    val value = new Document(document).sentences().toList.flatMap { a =>
      val words = a.words().toList
      val tags = a.posTags().toList
      (words zip tags).toMap.map { a =>
        val newWord = morphology.lemma(a._1, a._2)
        val addedWoed = if (newWord.length > 3) {
          newWord
        } else {
          ""
        }
        string.append(addedWoed + " ")
      }
    }
    removeDoubleSpaces(string.toString())
  }
}