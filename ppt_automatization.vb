
#If VBA7 Then
    Public Declare PtrSafe Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As LongPtr) 'For 64 Bit Systems
#Else
    Public Declare Sub Sleep Lib "kernel32" (ByVal dwMilliseconds As Long) 'For 32 Bit Systems
#End If

Option Explicit

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
        With textRange.Characters(boldStartPos + Len(boldStart), boldEndPos - 1).font
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
        With textRange.Characters(italicStartPos + 1, italicEndPos - 2).font
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
    Dim shapeZOrder As New Scripting.Dictionary

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
    Debug.Print "Indexing files..."
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
    Debug.Print "Iterating slides..."
    Dim slide As PowerPoint.slide
    For Each slide In ActivePresentation.Slides
        Debug.Print "Iterating shapes in slide " & slide.SlideNumber
        Dim shape As PowerPoint.shape
        For Each shape In slide.Shapes
            If left(shape.Name, 2) = "#+" Then ' if the object is to be replaced
                Debug.Print "Checking shape " & shape.Name
                idx = IndexOf(fileNames, Mid(shape.Name, 3)) ' get the index of the file by name
                If idx <> -1 Then ' if file is found
                      Debug.Print "a matching file was found "
                    ' check if it's a picture or a text file
                    If Right(fileNames.Item(idx), 3) = "png" Or Right(fileNames.Item(idx), 3) = "jpg" Then
                        Debug.Print "11111 Its a PNG 11111"
                        If shape.Type = msoPicture Or shape.Type = msoLinkedPicture Then ' if shape is a picture
                            
                            ' Before deleting the old shape, store its Z-Order
                            If Not shapeZOrder.Exists(shape.Name) Then
                                shapeZOrder.Add shape.Name, shape.ZOrderPosition
                            End If
                            
                            ' keep old picture properties
                            Dim oldCropTop As Single: oldCropTop = shape.PictureFormat.CropTop
                            Dim oldCropBottom As Single: oldCropBottom = shape.PictureFormat.CropBottom
                            Dim oldCropLeft As Single: oldCropLeft = shape.PictureFormat.CropLeft
                            Dim oldCropRight As Single: oldCropRight = shape.PictureFormat.CropRight
                            
                            ' remove cropping before deleting
                            With shape.PictureFormat
                                .CropTop = 0
                                .CropBottom = 0
                                .CropLeft = 0
                                .CropRight = 0
                            End With
                            
                            ' image size without cropping
                            Dim oldName As String: oldName = shape.Name
                            Dim oldLeft As Single: oldLeft = shape.left
                            Dim oldTop As Single: oldTop = shape.top
                            Dim oldWidth As Single: oldWidth = shape.width
                            Dim oldHeight As Single: oldHeight = shape.height
                            
                            shape.Delete
                            
                            ' add new picture
                            Dim newShape As PowerPoint.shape
                            Set newShape = slide.Shapes.AddPicture(filePaths.Item(idx), _
                                    msoFalse, msoTrue, oldLeft, oldTop, oldWidth, oldHeight)
                            
                            ' restore cropping
                            With newShape.PictureFormat
                                .CropTop = oldCropTop
                                .CropBottom = oldCropBottom
                                .CropLeft = oldCropLeft
                                .CropRight = oldCropRight
                            End With
                            
                            ' After adding the new shape, restore its Z-Order
                            If shapeZOrder.Exists(newShape.Name) Then
                                Do While newShape.ZOrderPosition < shapeZOrder(newShape.Name)
                                    newShape.ZOrder msoBringForward
                                Loop
                                Do While newShape.ZOrderPosition > shapeZOrder(newShape.Name)
                                    newShape.ZOrder msoSendBackward
                                Loop
                            End If
                            
                            newShape.Name = oldName
                            
                            ' sleep for 50 milliseconds
                            Sleep 50
                            Debug.Print "----FILE WAS REPLACED ----- "
                        End If
                    ElseIf Right(fileNames.Item(idx), 3) = "txt" Then
                        If shape.Type = msoTextBox Then ' if shape is a text box
                            ' read the content of the text file and replace the text box content
                            Dim textStream As textStream
                            Set textStream = fso.OpenTextFile(filePaths.Item(idx), ForReading)
                            ApplyTextStyle shape.TextFrame.textRange, textStream.ReadAll
                            textStream.Close
                            
                            ' sleep for 50 milliseconds
                            Sleep 50
                        End If
                    End If
                End If
            End If
        Next shape
    Next slide
End Sub





