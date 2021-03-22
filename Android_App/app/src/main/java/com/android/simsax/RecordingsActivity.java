package com.android.simsax;

import android.content.Intent;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class RecordingsActivity extends AppCompatActivity{
    private File[] files;
    private RecyclerView audioList;
    private AudioListAdapter audioListAdapter;
    private MediaPlayer mediaPlayer = null;
    private boolean isPlaying = false;
    private boolean paused = false;
    private ExampleRow thisFile = null;
    private ArrayList<ExampleRow> exampleList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recordings);
        Toolbar toolbar = findViewById(R.id.toolbarRecordings);
        toolbar.setTitleTextColor(Color.WHITE);
        setSupportActionBar(toolbar);
        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setDisplayShowHomeEnabled(true);
        audioList = findViewById(R.id.audio_list_view1);
        mediaPlayer = new MediaPlayer();

        String filesPath = getExternalFilesDir("/").getAbsolutePath();
        File directory = new File(filesPath);
        files = directory.listFiles();

        exampleList = new ArrayList<>();

        for (File file : files) {
            exampleList.add(new ExampleRow(R.drawable.ic_baseline_play_circle_24, file.getName(), file));
        }

        audioListAdapter = new AudioListAdapter(this, exampleList);
        audioList.setHasFixedSize(true);
        audioList.setLayoutManager(new LinearLayoutManager(getApplicationContext()));
        audioList.setAdapter(audioListAdapter);

        audioListAdapter.setOnItemClickListener(new AudioListAdapter.OnItemClickListener() {
            @Override
            public void onItemClick(int position) {
                elementClick(exampleList.get(position), position);
                audioListAdapter.notifyItemChanged(position);
            }

            @Override
            public void onShareClick(int position) {
                shareClick(exampleList.get(position));
            }
        });

    }

    public void elementClick(ExampleRow itemPlayed, int position) {
        if (isPlaying && itemPlayed == thisFile) {
            pauseAudio(itemPlayed);
        } else {
            if (isPlaying) {
                int pos = 0;
                for (ExampleRow element : exampleList) {
                    if (element.getImageButton() == R.drawable.ic_baseline_pause_circle_24) {
                        element.setImageButton(R.drawable.ic_baseline_play_circle_24);
                        audioListAdapter.notifyItemChanged(pos);
                        break;
                    }
                    pos++;
                }

            }

            mediaPlayer.stop();
            mediaPlayer.reset();
            thisFile = itemPlayed;
            itemPlayed.setImageButton(R.drawable.ic_baseline_pause_circle_24);
            playAudio(itemPlayed, position);
        }
    }

    private void pauseAudio(ExampleRow itemPlayed) {
        paused = !paused;
        if (paused) {
            mediaPlayer.pause();
            itemPlayed.setImageButton(R.drawable.ic_baseline_play_circle_24);
        } else {
            mediaPlayer.start();
            itemPlayed.setImageButton(R.drawable.ic_baseline_pause_circle_24);
        }
    }


    private void playAudio(ExampleRow itemPlayed, int position) {
        try {
            mediaPlayer.setDataSource(itemPlayed.getFile().getAbsolutePath());
            mediaPlayer.prepare();
            mediaPlayer.start();
        } catch (IOException e) {
            e.printStackTrace();
        }

        mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                itemPlayed.setImageButton(R.drawable.ic_baseline_play_circle_24);
                audioListAdapter.notifyItemChanged(position);
                isPlaying = false;
            }
        });
        isPlaying = true;
    }

    public void shareClick(ExampleRow currentItem) {
        Intent shareIntent = new Intent();
        shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        shareIntent.setAction(Intent.ACTION_SEND);
        shareIntent.putExtra(Intent.EXTRA_STREAM, FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", currentItem.getFile()));
        shareIntent.setType("audio/mp3");
        startActivity(Intent.createChooser(shareIntent, getResources().getText(R.string.send_to)));
    }

}

