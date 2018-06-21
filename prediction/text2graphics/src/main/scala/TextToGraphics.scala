import java.awt.Color
import java.awt.Font
import java.awt.FontMetrics
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException
import javax.imageio.ImageIO

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object TextToGraphics {

  def wrapSentences(sens: Seq[Seq[String]], n: Int, weights: Seq[Seq[Double]]): Seq[Seq[(String,Double)]] = {
    
    val builder = new StringBuilder()
    var sentences = ArrayBuffer[Seq[(String,Double)]]()
    var wordsBuf = ArrayBuffer[(String,Double)]()
    var size = 0
    for ((words,ws) <- sens zip weights) {
      assert(words.length == ws.length)
      var idx = 0
      var size = 0
      wordsBuf = ArrayBuffer[(String,Double)]()
      while(idx < words.length) {
        if (size > n) {
          sentences += wordsBuf.toSeq
          wordsBuf = ArrayBuffer[(String,Double)]()
          size = 0
        }
        wordsBuf += ((words(idx),ws(idx)))
        size += words(idx).length + 1
        idx += 1
      }
      sentences += wordsBuf
    }
    sentences
  }

  def draw(sentences: Seq[Seq[(String,Double)]]) = {

    var img = new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB)
    var g2d = img.createGraphics
    val fontSize = 17
    val font = new Font("Arial", Font.PLAIN, fontSize)
    g2d.setFont(font)
    var fm = g2d.getFontMetrics
    val width = 300*2
    //fm.stringWidth(sentences.maxBy(s => fm.stringWidth(s.map(_._1).mkString(" "))).map(_._1).mkString(" "))+20
    val height = (225*1.8).toInt
    //fm.getHeight * (sentences.size + fontSize / 10)
    g2d.dispose()
    img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    g2d = img.createGraphics
    g2d.setRenderingHint(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY)
    g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g2d.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_QUALITY)
    g2d.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE)
    g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON)
    g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
    g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE)
    g2d.setFont(font)
    fm = g2d.getFontMetrics
    g2d.setColor(new Color(0,0,0))
    //g2d.setColor(new Color(255,255,255))
    g2d.fillRect(0,0,width,height)

    var y = fm.getAscent
    var x = 3
    for ((sen, i) <- sentences.zipWithIndex) {
      x = 3
      var firstWord = true
      for ((word, weight) <- sen) {
        val c = new Color(255,255,255)
        //val c = new Color(255-(255*weight).toInt,255,255-(255*weight).toInt)
        g2d.setColor(c)
        val toDraw = if (firstWord && !word.startsWith("*")) "   "+word else word
        g2d.drawString(toDraw, x, y + i*(fm.getHeight + 5))
        x += fm.stringWidth(toDraw+" ")
        firstWord = false
      }
    }
    g2d.dispose()
    img
  }

  def main(args: Array[String]) {

    val path = args(0)
    val weightsPath = args(1)
    val wrapN = args(2).toInt
    val outPath = args(3)

    new File(outPath).mkdirs()

    for (file <- new File(path).listFiles()) {
      val words = Source.fromFile(file).getLines().toSeq.map(s => ("* "+s).split("\\s+").toSeq)
      val weights = Source.fromFile(weightsPath + "/" + file.getName()).getLines().toSeq.map(w => ("0.0 "+w).split(" ").toSeq.map(_.toDouble))
      val sentences = wrapSentences(words, wrapN, weights=weights)
      val img = draw(sentences)
      val outFilePath = outPath + "/" + file.getName.split("\\.").dropRight(1).mkString(".") + ".png"
      ImageIO.write(img, "png", new File(outFilePath))
    }
  }
}

