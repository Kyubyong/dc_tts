package com.android.simsax;

import android.content.Intent;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.Toast;;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    Toolbar toolbar;
    ImageButton playButton;
    ImageButton flagButton;
    ProgressBar pb;
    boolean ita;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        toolbar = findViewById(R.id.main_toolbar);
        toolbar.setTitleTextColor(Color.WHITE);
        setSupportActionBar(toolbar);
        pb = findViewById(R.id.progressBar);
        pb.setVisibility(View.INVISIBLE);
        ita = true;
    }

    public void playButtonPressed(View v) throws IOException {
        playButton = (ImageButton) v;
        TextInputLayout til = findViewById(R.id.text_input);
        String msg = til.getEditText().getText().toString();
        if (!msg.isEmpty()) {
            ConnectionThread connect = new ConnectionThread(msg);
            connect.start();
            playButton.setVisibility(View.INVISIBLE);
            pb.setVisibility(View.VISIBLE);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.recordings:
                Intent i = new Intent(this, RecordingsActivity.class);
                startActivity(i);
                return true;
          //case R.id.settings:
          //    return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public void flagPressed(View view) {
        flagButton = (ImageButton) view;
        TextInputEditText text = findViewById(R.id.text_input_edit);
        ita = !ita;
        if (ita) {
            text.setHint("Scrivi qualcosa...");
            flagButton.setImageDrawable(getResources().getDrawable(R.drawable.ic_italy));
        } else {
            text.setHint("Write something...");
            flagButton.setImageDrawable(getResources().getDrawable(R.drawable.ic_united_kingdom));
        }
    }


    private class ConnectionThread extends Thread {
        private String msg;

        ConnectionThread(String msg) {
            this.msg = msg;
        }

        @Override
        public void run() {
            Socket s = null;
            try {
                s = new Socket();
                s.connect(new InetSocketAddress("188.152.15.134", 1234), 3000);
            }
            catch (IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast toast = Toast.makeText(getApplicationContext(), "Can't connect to the server", Toast.LENGTH_LONG);
                        toast.setGravity(Gravity.CENTER_VERTICAL, 0, -400);
                        toast.show();
                        pb.setVisibility(View.INVISIBLE);
                        playButton.setVisibility(View.VISIBLE);
                    }
                });
                return;
            }
            try {
                PrintWriter pr = new PrintWriter(s.getOutputStream(), true);
                if (ita) {
                    msg = "ITA" + msg;
                } else {
                    msg = "ENG" + msg;
                }
                pr.print(msg);
                Log.d("msg_sent", msg);
                pr.flush();
                s.setSoTimeout(120 * 1000); // 2 minutes max to let the server neural network do the feed forward
                receiveAudio(s);
                s.close();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        pb.setVisibility(View.INVISIBLE);
                        playButton.setVisibility(View.VISIBLE);
                    }
                });
            } catch (IOException e) {
                Log.d("ERROR", e.toString());
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast toast = Toast.makeText(getApplicationContext(), "The server is taking too long to respond", Toast.LENGTH_LONG);
                        toast.setGravity(Gravity.CENTER_VERTICAL, 0, -400);
                        toast.show();
                    }
                });
                return;
            }
        }

        private void receiveAudio(Socket s) throws IOException {
            DataInputStream in = new DataInputStream(new BufferedInputStream(s.getInputStream()));
            byte[] msgByte = new byte[10000000]; // 10 MB (it would be better if the server sent a header in order to calculate the length, but I was lazy)
            String directoryToStore = getExternalFilesDir("/").getAbsolutePath();

            SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy_MM_dd_hh_mm_ss");
            String timestamp = simpleDateFormat.format(new Date());
            File dstFile = new File(directoryToStore + "/" + timestamp + ".wav");
            FileOutputStream out = new FileOutputStream(dstFile);

            int len;
            while ((len = in.read(msgByte)) > 0) {
                out.write(msgByte, 0, len);
            }
            out.close();

            MediaPlayer mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(dstFile.getAbsolutePath());
            mediaPlayer.prepare();
            mediaPlayer.start();
        }
    }
}