package com.example.applaunchergoogle;

import android.content.ContentValues;
import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;
import android.widget.Toast;

import androidx.annotation.Nullable;

import java.sql.Statement;

public class DatabaseConnector extends SQLiteOpenHelper {

    static String name = "loopdloopaidb";
    static int version = 1;

    String createTableUserInfo = "CREATE TABLE IF NOT EXISTS \"user_info\" (\n" +
            "\t\"id\"\tINTEGER PRIMARY KEY AUTOINCREMENT,\n" +
            "\t\"username\"\tTEXT,\n" +
            "\t\"password\"\tTEXT,\n" +
            "\t\"email\"\tTEXT,\n" +
            "\t\"country\"\tTEXT,\n" +
            "\t\"dob\"\tTEXT,\n" +
            "\t\"gender\"\tTEXT\n" +
            ")";

    public DatabaseConnector(Context context) {
        super(context, name, null, version);
        System.out.println(createTableUserInfo);
        getWritableDatabase().execSQL(createTableUserInfo);
    }

    public void insertNewUser(ContentValues contentValues) {
        System.out.println("Kuch to bataoo!");
        System.out.println(contentValues);
        long pk = getWritableDatabase().insert("user_info","",contentValues);
        System.out.println(pk);
    }

    public boolean isLoginValid(String username, String password) {
        String sql = "SELECT COUNT(*) FROM user_info WHERE username='" + username + "' AND password='" + password + "'";
        SQLiteStatement statement = getReadableDatabase().compileStatement(sql);
        long l = statement.simpleQueryForLong();
        statement.close();

        if (l == 1){
            return true;
        }else {
            return false;
        }
    }

    @Override
    public void onCreate(SQLiteDatabase db) {

    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }
}
