package com.arvoreneural.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.os.Build
import android.os.IBinder

/**
 * Serviço em primeiro plano: mantém o processo vivo com a tela apagada ou em outro app,
 * para a osmose, o treino manual e as respostas da IA não pararem. Some sozinho quando não há trabalho.
 */
class TrainService : Service() {
    @Volatile private var alive = false
    private var worker: Thread? = null

    override fun onBind(i: Intent?): IBinder? = null

    override fun onStartCommand(i: Intent?, flags: Int, startId: Int): Int {
        createChannel()
        val n = build("Preparando…")
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIF_ID, n)
        }
        val pause = i?.action == ACTION_PAUSE
        synchronized(this) {
            if (worker?.isAlive == true) {
                if (pause) Thread { if (Engine.ready) Engine.osmose.enabled = false }.start()
                return START_NOT_STICKY
            }
            alive = true
            val w = Thread {
                Engine.init(applicationContext)
                if (pause) Engine.osmose.enabled = false
                val nm = getSystemService(NotificationManager::class.java)
                while (alive && Engine.wantsService()) {
                    try {
                        nm.notify(NOTIF_ID, build(Engine.osmose.status()))
                    } catch (e: Throwable) {
                    }
                    try {
                        Thread.sleep(5000)
                    } catch (e: InterruptedException) {
                    }
                }
                alive = false
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            w.isDaemon = true
            worker = w
            w.start()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        alive = false
        super.onDestroy()
    }

    private fun createChannel() {
        val ch = NotificationChannel(CHANNEL, "Treino da Árvore Neural", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    private fun build(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val pauseIntent = PendingIntent.getService(
            this, 1, Intent(this, TrainService::class.java).setAction(ACTION_PAUSE),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val act = Notification.Action.Builder(
            Icon.createWithResource(this, android.R.drawable.ic_media_pause), "Desligar osmose", pauseIntent
        ).build()
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("Árvore Neural")
            .setContentText(text.lineSequence().firstOrNull() ?: "")
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(open)
            .addAction(act)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    companion object {
        const val CHANNEL = "treino"
        const val NOTIF_ID = 7
        const val ACTION_PAUSE = "com.arvoreneural.app.PAUSE"
    }
}
