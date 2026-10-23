package com.arvoreneural.app.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer

class Node(
    val id: Long,
    val parentId: Long,
    val title: String,
    val content: String,
    val embedding: FloatArray,
    val createdAt: Long
)

class Point(val id: Long, val series: String, val value: Double, val ts: Long)

class Db(ctx: Context) : SQLiteOpenHelper(ctx, "arvore.db", null, 4) {

    override fun onCreate(d: SQLiteDatabase) {
        createChat(d)
        createModel(d)
        createCorpus(d)
        d.execSQL(
            "CREATE TABLE nodes(id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id INTEGER NOT NULL, " +
                "title TEXT NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL, created_at INTEGER NOT NULL)"
        )
        d.execSQL(
            "CREATE TABLE points(id INTEGER PRIMARY KEY AUTOINCREMENT, series TEXT NOT NULL, " +
                "value REAL NOT NULL, ts INTEGER NOT NULL)"
        )
    }

    private fun createCorpus(d: SQLiteDatabase) {
        d.execSQL("CREATE TABLE IF NOT EXISTS corpus(idx INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    }

    private fun createModel(d: SQLiteDatabase) {
        d.execSQL("CREATE TABLE IF NOT EXISTS model(idx INTEGER PRIMARY KEY, data BLOB NOT NULL)")
    }

    private fun createChat(d: SQLiteDatabase) {
        d.execSQL(
            "CREATE TABLE IF NOT EXISTS chat(id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "from_user INTEGER NOT NULL, text TEXT NOT NULL, ts INTEGER NOT NULL)"
        )
    }

    override fun onUpgrade(d: SQLiteDatabase, oldV: Int, newV: Int) {
        if (oldV < 2) createChat(d)
        if (oldV < 3) createModel(d)
        if (oldV < 4) createCorpus(d)
    }

    private fun toBlob(a: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 4)
        bb.asFloatBuffer().put(a)
        return bb.array()
    }

    private fun fromBlob(b: ByteArray): FloatArray {
        val fb = ByteBuffer.wrap(b).asFloatBuffer()
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    @Synchronized
    fun insertNode(parentId: Long, title: String, content: String, emb: FloatArray): Long {
        val cv = ContentValues()
        cv.put("parent_id", parentId)
        cv.put("title", title)
        cv.put("content", content)
        cv.put("embedding", toBlob(emb))
        cv.put("created_at", System.currentTimeMillis())
        return writableDatabase.insert("nodes", null, cv)
    }

    @Synchronized
    fun updateEmbedding(id: Long, emb: FloatArray) {
        val cv = ContentValues()
        cv.put("embedding", toBlob(emb))
        writableDatabase.update("nodes", cv, "id=?", arrayOf(id.toString()))
    }

    @Synchronized
    fun allNodes(): List<Node> {
        val out = ArrayList<Node>()
        val c = readableDatabase.rawQuery(
            "SELECT id,parent_id,title,content,embedding,created_at FROM nodes ORDER BY id", null
        )
        c.use {
            while (it.moveToNext()) {
                out.add(
                    Node(
                        it.getLong(0), it.getLong(1), it.getString(2),
                        it.getString(3), fromBlob(it.getBlob(4)), it.getLong(5)
                    )
                )
            }
        }
        return out
    }

    @Synchronized
    fun deleteNode(id: Long) {
        val w = writableDatabase
        w.beginTransaction()
        try {
            val c = w.rawQuery("SELECT parent_id FROM nodes WHERE id=?", arrayOf(id.toString()))
            val p = c.use { if (it.moveToFirst()) it.getLong(0) else 0L }
            w.execSQL("UPDATE nodes SET parent_id=? WHERE parent_id=?", arrayOf<Any>(p, id))
            w.delete("nodes", "id=?", arrayOf(id.toString()))
            w.setTransactionSuccessful()
        } finally {
            w.endTransaction()
        }
    }

    @Synchronized
    fun insertPoint(series: String, value: Double) {
        val cv = ContentValues()
        cv.put("series", series)
        cv.put("value", value)
        cv.put("ts", System.currentTimeMillis())
        writableDatabase.insert("points", null, cv)
    }

    @Synchronized
    fun clearNodes() {
        val w = writableDatabase
        w.delete("nodes", null, null)
        w.execSQL("DELETE FROM sqlite_sequence WHERE name='nodes'")
    }

    @Synchronized
    fun addChat(fromUser: Boolean, text: String) {
        val cv = ContentValues()
        cv.put("from_user", if (fromUser) 1 else 0)
        cv.put("text", text)
        cv.put("ts", System.currentTimeMillis())
        writableDatabase.insert("chat", null, cv)
    }

    @Synchronized
    fun chatHistory(limit: Int = 200): List<Pair<Boolean, String>> {
        val out = ArrayList<Pair<Boolean, String>>()
        val c = readableDatabase.rawQuery(
            "SELECT from_user,text FROM chat ORDER BY id DESC LIMIT ?", arrayOf(limit.toString())
        )
        c.use {
            while (it.moveToNext()) out.add((it.getInt(0) == 1) to it.getString(1))
        }
        out.reverse()
        return out
    }

    @Synchronized
    fun clearChat() {
        writableDatabase.delete("chat", null, null)
    }

    /** Pesos do transformer em pedaços de ~400 KB (o Android limita ~2 MB por linha lida). */
    @Synchronized
    fun saveModel(meta: ByteArray, w: FloatArray) {
        val d = writableDatabase
        d.beginTransaction()
        try {
            d.delete("model", null, null)
            val m = ContentValues()
            m.put("idx", -1)
            m.put("data", meta)
            d.insert("model", null, m)
            val chunk = 100_000
            var idx = 0
            var off = 0
            while (off < w.size) {
                val n = minOf(chunk, w.size - off)
                val bb = ByteBuffer.allocate(n * 4)
                bb.asFloatBuffer().put(w, off, n)
                val cv = ContentValues()
                cv.put("idx", idx)
                cv.put("data", bb.array())
                d.insert("model", null, cv)
                idx++
                off += n
            }
            d.setTransactionSuccessful()
        } finally {
            d.endTransaction()
        }
    }

    @Synchronized
    fun loadModelMeta(): ByteArray? {
        val c = readableDatabase.rawQuery("SELECT data FROM model WHERE idx=-1", null)
        return c.use { if (it.moveToFirst()) it.getBlob(0) else null }
    }

    @Synchronized
    fun loadModelWeights(into: FloatArray): Boolean {
        var off = 0
        val c = readableDatabase.rawQuery("SELECT data FROM model WHERE idx>=0 ORDER BY idx", null)
        c.use {
            while (it.moveToNext()) {
                val fb = ByteBuffer.wrap(it.getBlob(0)).asFloatBuffer()
                val n = fb.remaining()
                if (off + n > into.size) return false
                fb.get(into, off, n)
                off += n
            }
        }
        return off == into.size
    }

    @Synchronized
    fun clearModel() {
        writableDatabase.delete("model", null, null)
    }

    @Synchronized
    fun nodeCount(): Int {
        val c = readableDatabase.rawQuery("SELECT COUNT(*) FROM nodes", null)
        return c.use { if (it.moveToFirst()) it.getInt(0) else 0 }
    }

    @Synchronized
    fun corpusChars(): Int {
        val c = readableDatabase.rawQuery("SELECT COALESCE(SUM(LENGTH(text)),0) FROM corpus", null)
        return c.use { if (it.moveToFirst()) it.getInt(0) else 0 }
    }

    /** Acrescenta texto ao corpus de treino (pedaços de até 40 mil caracteres). Devolve quantos caracteres entraram. */
    @Synchronized
    fun addCorpus(text: String, limit: Int): Int {
        val room = limit - corpusChars()
        if (room <= 0) return 0
        val t = if (text.length > room) text.substring(0, room) else text
        val d = writableDatabase
        d.beginTransaction()
        try {
            val c = d.rawQuery("SELECT COALESCE(MAX(idx),-1) FROM corpus", null)
            var idx = c.use { if (it.moveToFirst()) it.getInt(0) + 1 else 0 }
            var off = 0
            while (off < t.length) {
                var end = minOf(off + 40_000, t.length)
                if (end < t.length && Character.isHighSurrogate(t[end - 1])) end--
                val cv = ContentValues()
                cv.put("idx", idx)
                cv.put("text", t.substring(off, end))
                d.insert("corpus", null, cv)
                idx++
                off = end
            }
            d.setTransactionSuccessful()
        } finally {
            d.endTransaction()
        }
        return t.length
    }

    @Synchronized
    fun corpusText(): String {
        val sb = StringBuilder()
        val c = readableDatabase.rawQuery("SELECT text FROM corpus ORDER BY idx", null)
        c.use {
            while (it.moveToNext()) sb.append(it.getString(0))
        }
        return sb.toString()
    }

    @Synchronized
    fun clearCorpus() {
        writableDatabase.delete("corpus", null, null)
    }

    @Synchronized
    fun points(series: String): List<Point> {
        val out = ArrayList<Point>()
        val c = readableDatabase.rawQuery(
            "SELECT id,series,value,ts FROM points WHERE series=? ORDER BY id", arrayOf(series)
        )
        c.use {
            while (it.moveToNext()) out.add(Point(it.getLong(0), it.getString(1), it.getDouble(2), it.getLong(3)))
        }
        return out
    }
}
