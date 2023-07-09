Option Explicit

Sub ReplaceObjects()

    Dim fso As New FileSystemObject
    Dim folder As folder
    Dim subfolder As folder
    Dim file As file
    Dim filePaths As New Collection
    Dim fileNames As New Collection

    ' 1. Prompt for a directory
    With Application.FileDialog(msoFileDialogFolderPicker)
        .Title = "Select Parent Directory"
        .AllowMultiSelect = False
        If .Show = -1 Then ' If OK is pressed
            Set folder = fso.GetFolder(.SelectedItems(1))
        Else
            Exit Sub
        End If
    End With

    ' 2. Index all files within the directory and its subdirectories
    For Each file In folder.Files
        filePaths.Add file.Path
        fileNames.Add file.Name
    Next file
    For Each subfolder In folder.SubFolders
        For Each file In subfolder.Files
            filePaths.Add file.Path
            fileNames.Add file.Name
        Next file
    Next subfolder

    ' 3. Iterate through all slides and objects
    Dim slide As slide
    Dim shape As Shape
    Dim idx As Integer
    For Each slide In ActivePresentation.Slides
        For Each shape In slide.Shapes
            If Left(shape.Name, 2) = "#+" Then ' if the object is to be replaced
                idx = fileNames.IndexOf(Mid(shape.Name, 3)) ' get the index of the file
                If idx <> -1 Then ' if file is found
                    If shape.Type = msoPicture Then ' if shape is a picture
                        ' delete the existing picture and add the new picture
                        Dim left As Single: left = shape.Left
                        Dim top As Single: top = shape.Top
                        Dim width As Single: width = shape.Width
                        Dim height As Single: height = shape.Height
                        shape.Delete
                        slide.Pictures.Insert(filePaths.Item(idx), left, top, width, height)
                    ElseIf shape.Type = msoTextBox Then ' if shape is a text box
                        ' read the content of the text file and replace the text box content
                        Dim textStream As TextStream
                        Set textStream = fso.OpenTextFile(filePaths.Item(idx), ForReading)
                        shape.TextFrame.TextRange.Text = textStream.ReadAll
                        textStream.Close
                    End If
                End If
            End If
        Next shape
    Next slide
End Sub
