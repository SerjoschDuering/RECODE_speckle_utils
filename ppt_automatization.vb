Function IndexOf(col As Collection, val As Variant) As Integer
    Dim i As Integer
    For i = 1 To col.Count
        If col(i) = val Then
            IndexOf = i
            Exit Function
        End If
    Next i
    IndexOf = -1
End Function

Sub ApplyTextStyle(ByRef textRange As textRange, ByVal txt As String)

    ' Define start and end tags for bold and italic text
    Dim boldStart As String: boldStart = "**"
    Dim boldEnd As String: boldEnd = "**"
    Dim italicStart As String: italicStart = "__"
    Dim italicEnd As String: italicEnd = "__"
    
    ' Create variables for positions
    Dim boldStartPos As Integer
    Dim boldEndPos As Integer
    Dim italicStartPos As Integer
    Dim italicEndPos As Integer

    ' Apply bold formatting
    boldStartPos = InStr(txt, boldStart)
    boldEndPos = InStr(txt, boldEnd)
    While boldStartPos > 0 And boldEndPos > 0
        With textRange.Characters(boldStartPos + Len(boldStart), boldEndPos - 1).Font

            .Bold = msoTrue
        End With
        txt = Replace(txt, boldStart, "", 1, 1)
        txt = Replace(txt, boldEnd, "", 1, 1)
        boldStartPos = InStr(txt, boldStart)
        boldEndPos = InStr(txt, boldEnd)
    Wend

    ' Apply italic formatting
    italicStartPos = InStr(txt, italicStart)
    italicEndPos = InStr(txt, italicEnd)
    While italicStartPos > 0 And italicEndPos > 0
        With textRange.Characters(italicStartPos + 1, italicEndPos - 2).Font
            .Italic = msoTrue
        End With
        txt = Replace(txt, italicStart, "", 1, 1)
        txt = Replace(txt, italicEnd, "", 1, 1)
        italicStartPos = InStr(txt, italicStart)
        italicEndPos = InStr(txt, italicEnd)
    Wend

    ' Assign the modified text to the text range
    textRange.Text = txt
End Sub



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
    Dim idx As Integer
    For Each slide In ActivePresentation.Slides
        For Each shape In slide.Shapes
            If left(shape.Name, 2) = "#+" Then ' if the object is to be replaced
                idx = IndexOf(fileNames, Mid(shape.Name, 3)) ' get the index of the file by name
                If idx <> -1 Then ' if file is found
                    ' check if it's a picture or a text file
                    If Right(fileNames.Item(idx), 3) = "png" Or Right(fileNames.Item(idx), 3) = "jpg" Then
                        If shape.Type = msoPicture Then ' if shape is a picture
                            ' keep old picture properties
                            Dim oldName As String: oldName = shape.Name
                            Dim oldLeft As Single: oldLeft = shape.left
                            Dim oldTop As Single: oldTop = shape.top
                            Dim oldWidth As Single: oldWidth = shape.width
                            Dim oldHeight As Single: oldHeight = shape.height
                            shape.Delete
                            ' add new picture
                            Dim newShape As shape
                            Set newShape = slide.Shapes.AddPicture(filePaths.Item(idx), _
                                    msoFalse, msoTrue, oldLeft, oldTop, oldWidth, oldHeight)
                            newShape.Name = oldName
                        End If
                    ElseIf Right(fileNames.Item(idx), 3) = "txt" Then
                        If shape.Type = msoTextBox Then ' if shape is a text box
                            ' read the content of the text file and replace the text box content
                            Dim textStream As textStream
                            Set textStream = fso.OpenTextFile(filePaths.Item(idx), ForReading)
                            ApplyTextStyle shape.TextFrame.textRange, textStream.ReadAll
                            textStream.Close
                        End If
                    End If
                End If
            End If
        Next shape
    Next slide
End Sub


