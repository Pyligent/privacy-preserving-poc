package com.example.applaunchergoogle;

import android.content.ContentValues;
import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;

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

    public DatabaseConnector(@Nullable Context context) {
        super(context, name, null, version);

        getWritableDatabase().execSQL(createTableUserInfo);
    }

    public void insertNewUser(ContentValues contentValues) {
        getWritableDatabase().insert("user_info","",contentValues);
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
